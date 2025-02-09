#!/usr/bin/env python3

import argparse
from copy import deepcopy
import sys

import numpy as np

from atom_calibration.collect import patterns
from atom_core.dataset_io import addNoiseToInitialGuess, filterCollectionsFromDataset, loadResultsJSON
from atom_core.utilities import atomError, createLambdaExpressionsForArgs

def skew_sym_matrix(vector):

    skew_symmetric_matrix = np.zeros((3,3))

    skew_symmetric_matrix[0,1] = -vector[2]
    skew_symmetric_matrix[0,2] = vector[1]
    skew_symmetric_matrix[1,0] = vector[2]
    skew_symmetric_matrix[1,2] = -vector[0]
    skew_symmetric_matrix[2,0] = -vector[1]
    skew_symmetric_matrix[2,1] = vector[0]

    return skew_symmetric_matrix   

def omega(vector):

    M = np.zeros((4,4))
    M[1:,1:] = -skew_sym_matrix(vector)
    M[1:, 0] = vector
    M[0, 1:] = -vector
    
    return M

def normalize_quaternion(quaternion):
    
    if quaternion[0] < 0:
        quaternion = -1 * quaternion
    
    normalized_quaternion = quaternion/np.linalg.norm(quaternion)

    return normalized_quaternion

def rk4_imu_integration(imu_data_0, imu_data_1):
    # This function receives two imu "data points" and integrates the angular velocity and linear acceleration to calculate the angular and linear displacements between these two points.

    # Get delta_t
    t_0 = imu_data_0["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_0["header"]["stamp"]["nsecs"]
    t_1 = imu_data_1["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_1["header"]["stamp"]["nsecs"]
    delta_t = t_1 - t_0

    initial_delta_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    # Orientation integration
    ang_vel_0 = np.array([*imu_data_0["angular_velocity"].values()])
    ang_vel_1 = np.array([*imu_data_1["angular_velocity"].values()])

    d_ang_vel = (ang_vel_1 - ang_vel_0)/delta_t

    k1 = delta_t * (0.5 * np.matmul(omega(ang_vel_0), initial_delta_orientation))
    k2 = delta_t * (0.5 * np.matmul(omega(ang_vel_0 + 0.5*d_ang_vel*delta_t), normalize_quaternion(initial_delta_orientation + 0.5*k1)))
    k3 = delta_t * (0.5 * np.matmul(omega(ang_vel_0 + 0.5*d_ang_vel*delta_t), normalize_quaternion(initial_delta_orientation + 0.5*k2)))
    k4 = delta_t * (0.5 * np.matmul(omega(ang_vel_0 + d_ang_vel*delta_t), normalize_quaternion(initial_delta_orientation + k3)))
    
    delta_orientation = normalize_quaternion(initial_delta_orientation + (k1 + 2*k2 + 2*k3 + k4)/6.0)
    
    return delta_orientation

def main():
    ########################################
    # ARGUMENT PARSER #
    ########################################

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-json", "--json_file", type=str, required=True, help="Json file containing input dataset.")
    ap.add_argument("-imu", "--imu_name", help="The name of the IMU sensor.", type=str, required=True)
    ap.add_argument("-csf", "--collection_selection_function", default=None, type=str, help="A string to be evaluated into a lambda function that receives a collection name as input and returns True or False to indicate if the collection should be loaded (and used in the optimization). The Syntax is lambda name: f(x), where f(x) is the function in python language. Example: lambda name: int(name) > 5 , to load only collections 6, 7, and onward.")
    ap.add_argument("-uic", "--use_incomplete_collections", action="store_true", default=False, help="Remove any collection which does not have a detection for all sensors.", )
    
    # Roslaunch adds two arguments (__name and __log) that break our parser. Lets remove those.
    arglist = [x for x in sys.argv[1:] if not x.startswith("__")]
    # these args have the selection functions as strings
    args_original = vars(ap.parse_args(args=arglist))
    args = createLambdaExpressionsForArgs(args_original)  # selection functions are now lambdas

    json_file = args['json_file']
    imu_name = args["imu_name"]
    collection_selection_function = args["collection_selection_function"]
    
    # Read dataset file
    dataset, json_file = loadResultsJSON(json_file, collection_selection_function)

    dataset_ground_truth = deepcopy(dataset)  # make a copy before adding noise

    # ---------------------------------------
    # --- Define selected collection key.
    # ---------------------------------------
    # We only need to get one collection because optimized transformations are static, which means they are the same for all collections. Let's select the first key in the dictionary and always get that transformation.
    selected_collection_key = list(dataset["collections"].keys())[0]
    print("Selected collection key is " + str(selected_collection_key))

    # ---------------------------------------
    # --- Implementation
    # ---------------------------------------
    
    # For each collection, get a list of all IMU data from continuous_sensor_data from the previous collection to the next 
    for collection_key, collection in dataset["collections"].items():
        
        collection_stamp = (collection["data"][imu_name]["header"]["stamp"]["secs"], collection["data"][imu_name]["header"]["stamp"]["nsecs"])

        tmp_checkpoint = 0 # Here to avoid iterating over the same datapoints
        for i in range(tmp_checkpoint, len(dataset["continuous_sensor_data"][imu_name])):

            if i == 0:
                initial_orientation = np.array([*dataset["continuous_sensor_data"][imu_name][i]["orientation"].values()])
                tmp_q = initial_orientation
                continue

            data_0 = dataset["continuous_sensor_data"][imu_name][i-1]
            data_1 = dataset["continuous_sensor_data"][imu_name][i]

            delta_q = rk4_imu_integration(
                imu_data_0=data_0,
                imu_data_1=data_1
            )
            print(delta_q)


            # Get new orientation
            tmp_q = np.array([
            delta_q[0] * tmp_q[0] - delta_q[1] * tmp_q[1] - delta_q[2] * tmp_q[2] - delta_q[3] * tmp_q[3],  # w
            
            delta_q[0] * tmp_q[1] + delta_q[1] * tmp_q[0] + delta_q[2] * tmp_q[3] - delta_q[3] * tmp_q[2],  # x
            
            delta_q[0] * tmp_q[2] - delta_q[1] * tmp_q[3] + delta_q[2] * tmp_q[0] + delta_q[3] * tmp_q[1],  # y
            
            delta_q[0] * tmp_q[3] + delta_q[1] * tmp_q[2] - delta_q[2] * tmp_q[1] + delta_q[3] * tmp_q[0]   # z
            ])

            # if tmp_q[0] < 0:
            #     tmp_q = -1 * tmp_q
            
            if i == 1:
                print(f'Orientation at i == 1: {tmp_q}')

            if (dataset["continuous_sensor_data"][imu_name][i]["header"]["stamp"]["secs"], dataset["continuous_sensor_data"][imu_name][i]["header"]["stamp"]["nsecs"]) ==  collection_stamp:
                tmp_checkpoint = i
                final_orientation = tmp_q
                print(f'Orientation at collection {collection_key}: {final_orientation}')


if __name__ == "__main__":
    main()