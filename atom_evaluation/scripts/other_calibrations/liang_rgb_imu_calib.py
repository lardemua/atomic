#!/usr/bin/env python3

"""
Implementation of an ATOM-compatible alternative RGB-IMU calibration method described by Liang et. al (2024
"""

import argparse
from copy import deepcopy
from colorama import Fore
import numpy as np
import cv2
from prettytable import PrettyTable
from atom_calibration.collect import patterns
import tf

from atom_core.dataset_io import filterCollectionsFromDataset, loadResultsJSON
from atom_core.atom import getTransform, getChain
from atom_core.geometry import traslationRodriguesToTransform
from atom_core.naming import generateKey
from atom_core.transformations import compareTransforms
from atom_core.utilities import atomError, compareAtomTransforms


def getCameraIntrinsicsFromDataset(dataset, camera):
    # Camera intrinsics (from the dataset) needed to calculate B
    K = np.zeros((3, 3), np.float32)
    D = np.zeros((5, 1), np.float32)
    K[0, :] = dataset['sensors'][camera]['camera_info']['K'][0:3]
    K[1, :] = dataset['sensors'][camera]['camera_info']['K'][3:6]
    K[2, :] = dataset['sensors'][camera]['camera_info']['K'][6:9]
    D[:, 0] = dataset['sensors'][camera]['camera_info']['D'][0:5]

    height = dataset['sensors'][camera]['camera_info']['height']
    width = dataset['sensors'][camera]['camera_info']['width']
    image_size = (height, width)

    return K, D, image_size

def getPatternConfig(dataset, pattern):
    # Pattern configs
    nx = dataset['calibration_config']['calibration_patterns'][pattern]['dimension']['x']
    ny = dataset['calibration_config']['calibration_patterns'][pattern]['dimension']['y']
    square = dataset['calibration_config']['calibration_patterns'][pattern]['size']
    inner_square = dataset['calibration_config']['calibration_patterns'][pattern]['inner_size']
    objp = np.zeros((nx * ny, 3), np.float32)
    # set of coordinates (w.r.t. the pattern frame) of the corners
    objp[:, :2] = square * np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    return nx, ny, square, inner_square, objp

def main():

    ########################################
    # ARGUMENT PARSER #
    ########################################

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file", type=str, required=True,
                    help="Json file containing input dataset.")
    ap.add_argument("-csf", "--collection_selection_function", default=None, type=str,
                    help="A string to be evaluated into a lambda function that receives a collection name as input and "
                    "returns True or False to indicate if the collection should be loaded (and used in the "
                    "optimization). The Syntax is lambda name: f(x), where f(x) is the function in python "
                    "language. Example: lambda name: int(name) > 5 , to load only collections 6, 7, and onward.")
    ap.add_argument("-c", "--camera", help="Camera sensor name.", type=str, required=True)
    ap.add_argument("-iln", "--imu_link_name", help="The name of the IMU link.", type=str, required=True)    
    ap.add_argument("-p", "--pattern", help="Pattern to be used for calibration.", type=str, required=True)
    ap.add_argument("-uic", "--use_incomplete_collections", action="store_true", default=False, help="Remove any collection which does not have a detection for all sensors.", )
    ap.add_argument("-ctgt", "--compare_to_ground_truth", action="store_true", help="If the system being calibrated is simulated, directly compare the TFs to the ground truth.")
    
    args = vars(ap.parse_args())

    json_file = args['json_file']
    collection_selection_function = args['collection_selection_function']
    imu_link_name = args["imu_link_name"]
    camera = args['camera']
    pattern = args['pattern']

    # Read dataset file
    dataset, json_file = loadResultsJSON(json_file, collection_selection_function)
    args['remove_partial_detections'] = True
    dataset = filterCollectionsFromDataset(dataset, args)
    
    dataset_ground_truth = deepcopy(dataset)  # make a copy before adding noise
    dataset_initial = deepcopy(dataset)  # store initial values

    # ---------------------------------------
    # --- Define selected collection key.
    # ---------------------------------------
    # We only need to get one collection because optimized transformations are static, which means they are the same for all collections. Let's select the first key in the dictionary and always get that transformation.
    selected_collection_key = list(dataset["collections"].keys())[0]
    print("Selected collection key is " + str(selected_collection_key))

    # ---------------------------------------
    # Verifications
    # ---------------------------------------

    # Check that the camera has rgb modality
    if not dataset['sensors'][args['camera']]['modality'] == 'rgb':
        atomError('Sensor ' + args['camera'] + ' is not of rgb modality.')

    # ---------------------------------------
    # Pattern configuration
    # ---------------------------------------
    nx = dataset['calibration_config']['calibration_patterns'][args['pattern']
                                                               ]['dimension']['x']
    ny = dataset['calibration_config']['calibration_patterns'][args['pattern']
                                                               ]['dimension']['y']
    square = dataset['calibration_config']['calibration_patterns'][args['pattern']]['size']
    pts_3d = np.zeros((nx * ny, 3), np.float32)
    # set of coordinates (w.r.t. the pattern frame) of the corners
    pts_3d[:, :2] = square * np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    # print(pts_3d)
    number_of_corners = int(nx) * int(ny)

    # ---------------------------------------
    # --- Get intrinsic data for the camera
    # ---------------------------------------
    # Source sensor
    K = np.zeros((3, 3), np.float32)
    D = np.zeros((5, 1), np.float32)
    K[0, :] = dataset['sensors'][args['camera']]['camera_info']['K'][0:3]
    K[1, :] = dataset['sensors'][args['camera']]['camera_info']['K'][3:6]
    K[2, :] = dataset['sensors'][args['camera']]['camera_info']['K'][6:9]
    D[:, 0] = dataset['sensors'][args['camera']]['camera_info']['D'][0:5]


    # ---------------------------------------
    # --- Implementation
    # ---------------------------------------

    # Calculate camera-to-pattern tf (c_T_p/H_cw in the paper) for each collection
    c_T_p_lst = [] # list of tuples (collection, camera to pattern 4x4 transforms)

    for collection_key, collection in dataset['collections'].items():

        # Pattern not detected by sensor in collection
        if not collection['labels'][args['pattern']][args['camera']]['detected']:
            continue        
        
        # First, I need to initialize a board object
        board_size = {'x': nx, 'y': ny}
        inner_length = dataset['calibration_config']['calibration_patterns'][
            args['pattern']]['inner_size']
        dictionary = dataset['calibration_config']['calibration_patterns'][
            args['pattern']]['dictionary']
        pattern = patterns.CharucoPattern(board_size, square, inner_length, dictionary)

        # Build a numpy array with the charuco corners
        corners = np.zeros(
            (len(collection['labels'][args['pattern']][args['camera']]['idxs']), 1, 2), dtype=float)
        ids = list(range(0, len(collection['labels'][args['pattern']][args['camera']]['idxs'])))
        for idx, point in enumerate(collection['labels'][args['pattern']][args['camera']]['idxs']):
            corners[idx, 0, 0] = point['x']
            corners[idx, 0, 1] = point['y']
            ids[idx] = point['id']

        # Find pose of the camera w.r.t the chessboard
        np_ids = np.array(ids, dtype=int)
        rvec, tvec = None, None
        _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(np.array(corners, dtype=np.float32),
                                                             np_ids, pattern.board,
                                                             K, D, rvec, tvec)
        
        # Convert to 4x4 transform and add to list
        c_T_p = traslationRodriguesToTransform(tvec, rvec)
        c_T_p_lst.append((collection_key, c_T_p))
        
    
    # Get the interframe c_T_p (H_cij in the paper)
    # These will be stored in a dictionary where the key is name of the collections (i-j)
    interframe_c_T_p_dict = {}

    for i in range(len(c_T_p_lst) - 1):
        j = i+1        
        key_name = str(c_T_p_lst[i][0]) + '-' + str(c_T_p_lst[j][0]) # Get the name of the key for the dict
        c_T_p_i = c_T_p_lst[i][1]
        c_T_p_j_inv = np.linalg.inv(c_T_p_lst[j][1])
        
        # Equation 17 from the original paper states that the tranformation matrix of the camera from collection i to collection j is equal to the c_T_p in collection i multiplied by its inverse in collection j
        interframe_c_T_p = np.dot(c_T_p_i, c_T_p_j_inv) 
        
        interframe_c_T_p_dict[key_name] = interframe_c_T_p # Save to the dict

    

if __name__ == "__main__":
    main()