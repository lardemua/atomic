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
import tf
import math

from atom_core.dataset_io import filterCollectionsFromDataset, loadResultsJSON
from atom_core.atom import getTransform, getChain
from atom_core.geometry import traslationRodriguesToTransform
from atom_core.naming import generateKey
from atom_core.transformations import compareTransforms
from atom_core.utilities import atomError, compareAtomTransforms

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



if __name__ == "__main__":
    main()