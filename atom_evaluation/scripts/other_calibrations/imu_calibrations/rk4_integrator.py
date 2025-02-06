#!/usr/bin/env python3

import argparse
from copy import deepcopy
import sys

import numpy as np

from atom_calibration.collect import patterns
from atom_core.dataset_io import addNoiseToInitialGuess, filterCollectionsFromDataset, loadResultsJSON
from atom_core.utilities import atomError, createLambdaExpressionsForArgs


def generate_skew_symmetric_4x4_matrix_from_vector(vector):

    skew_symmetric_matrix = np.zeros((4,4))
    
    skew_symmetric_matrix[0,1] = -vector[0]
    skew_symmetric_matrix[0,2] = -vector[1]
    skew_symmetric_matrix[0,3] = -vector[2]

    skew_symmetric_matrix[1,0] = vector[0]
    skew_symmetric_matrix[1,2] = vector[2]
    skew_symmetric_matrix[1,3] = -vector[1]

    skew_symmetric_matrix[2,0] = vector[1]
    skew_symmetric_matrix[2,1] = -vector[2]
    skew_symmetric_matrix[2,3] = vector[0]
    
    skew_symmetric_matrix[3,0] = vector[2]
    skew_symmetric_matrix[3,1] = vector[1]
    skew_symmetric_matrix[3,2] = -vector[0]

    return skew_symmetric_matrix



def rk4_imu_integration(imu_data_0, imu_data_1, initial_orientation):
    # This function receives two imu "data points" and integrates the angular velocity and linear acceleration to calculate the angular and linear displacements between these two points.

    # Get delta_t
    t_0 = imu_data_0["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_0["header"]["stamp"]["nsecs"]
    t_1 = imu_data_1["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_1["header"]["stamp"]["nsecs"]
    delta_t = t_1 - t_0

    # initial_orientatio    n = np.array([*initial_orientation.values()]).T

    # Orientation integration
    omega_0 = np.array([*imu_data_0["angular_velocity"].values()])
    omega_1 = np.array([*imu_data_1["angular_velocity"].values()])


    q1 = initial_orientation
    omega_m_0 = generate_skew_symmetric_4x4_matrix_from_vector(omega_0) 
    k1 = (1/2) * (omega_m_0 @ q1)

    q2 = initial_orientation + delta_t*(1/2)*k1
    omega_m_halfway = generate_skew_symmetric_4x4_matrix_from_vector((omega_0 + omega_1)/2)
    k2 = (1/2) * (omega_m_halfway @ q2)

    q3 = initial_orientation + delta_t*(1/2)*k2
    k3 = (1/2) * (omega_m_halfway @ q3)

    q4 = initial_orientation + delta_t*k3
    omega_m_1 = generate_skew_symmetric_4x4_matrix_from_vector(omega_1)
    k4 = (1/2) * (omega_m_1 @ q4)

    final_orientation = initial_orientation + delta_t * ((k1/6) + (k2/3) + (k3/3) + (k4/6))

    return final_orientation

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
        imu_seq = collection["data"][imu_name]["header"]["seq"]

        tmp_checkpoint = 0 # Here to avoid iterating over the same datapoints
        imu_data_keys = []
        for i in range(tmp_checkpoint, len(dataset["continuous_sensor_data"][imu_name])):

            if i == 0:
                initial_orientation = np.array([*dataset["continuous_sensor_data"][imu_name][i]["orientation"].values()])
                continue
            
            data_0 = dataset["continuous_sensor_data"][imu_name][i-1]
            data_1 = dataset["continuous_sensor_data"][imu_name][i]

            final_orientation = rk4_imu_integration(
                imu_data_0=data_0,
                imu_data_1=data_1,
                initial_orientation=initial_orientation
            )

            initial_orientation = final_orientation

            if dataset["continuous_sensor_data"][imu_name][i]["header"]["seq"] ==  imu_seq:
                tmp_checkpoint = i
                print(final_orientation)



if __name__ == "__main__":
    main()