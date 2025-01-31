#!/usr/bin/env python
"""
Reads a set of data and labels from a group of sensors in a json file and calibrates the poses of these sensors.
"""

# stdlib

# 3rd-party
from copy import deepcopy
import math
import os
import re
import subprocess
from statistics import mean
from signal import setitimer, signal, SIGALRM, ITIMER_REAL
from prettytable import PrettyTable

import readchar
import rospkg
from colorama import Fore, Style
import numpy as np

# 3rd-party
from rospy_message_converter import message_converter
from pynput import keyboard
import yaml

from tf.transformations import quaternion_matrix, euler_from_matrix, euler_from_quaternion

from atom_core.naming import generateKey
from atom_core.system import execute, removeColorsFromText

# -------------------------------------------------------------------------------
# --- FUNCTIONS
# -------------------------------------------------------------------------------


def assertSensorModality(dataset, sensor, modality):

    if sensor not in dataset['sensors'].keys():
        atomError('Sensor' + Fore.BLUE + sensor + Style.RESET_ALL + ' does not exist in the dataset')

    if not dataset['sensors'][sensor]['modality'] == modality:
        atomError('Sensor ' + Fore.BLUE + sensor + Style.RESET_ALL + ' has modality ' + Fore.GREEN +
                  dataset['sensors'][sensor]['modality'] + Style.RESET_ALL + ' (it should be ' +
                  Fore.GREEN + modality + Style.RESET_ALL + ')')


def compareAtomTransforms(transform_1, transform_2):
    """Compares two different transformations and returns the following metrics:
    Args:
        transform_1: transformation 1
        transform_2: transformation 2
    """


    # Create a 4x4 transformation for transform_1
    t1 = quaternion_matrix(transform_1['quat'])
    t1[0:3, 3] = transform_1['trans']

    # Create a 4x4 transformation for transform_2
    t2 = quaternion_matrix(transform_2['quat'])
    t2[0:3, 3] = transform_2['trans']

    v = t2[0:3, 3] - t1[0:3, 3]

    # Method: We will use the following method. If T1 and T2 are the same, then multiplying one by the inverse of the other will produce and identity matrix, with zero translation and rotation. So we will do the multiplication and then evaluation of the amount of rotation and translation in the resulting matrix.
    # print('Comparing \nt1= ' + str(t1) + ' \n\nt2=' + str(t2))

    t_delta = np.dot(np.linalg.inv(t1), t2)
    # print('t_delta = ' + str(t_delta))

    rotation_delta = t_delta[0:3, 0:3]
    # roll, pitch, yaw = euler_from_matrix(rotation_delta)
    translation_delta = t_delta[0:3, 3]

    euler_angles_init = euler_from_quaternion(transform_1['quat'])
    euler_angles_final = euler_from_quaternion(transform_2['quat'])


    deuler = np.subtract(euler_angles_final,euler_angles_init)
    rotation_error = np.linalg.norm(deuler)

    # global metrics
    # translation_error = float(abs(np.average(translation_delta)))
    translation_error = np.linalg.norm(translation_delta)
    # rotation_error = np.linalg.norm([roll, pitch, yaw])


    # print('translation error = ' + str(translation_error))
    # print('rotation error = ' + str(rotation_error))

    return translation_error, rotation_error


def getNumberQualifier(n, unit='meters'):

    if unit == 'meters':
        if n < 0.001:
            return Fore.LIGHTGREEN_EX + '{:.5f}'.format(n) + Style.RESET_ALL
        elif n < 0.01:
            return Fore.GREEN + '{:.5f}'.format(n) + Style.RESET_ALL
        elif n < 0.05:
            return Fore.YELLOW + '{:.5f}'.format(n) + Style.RESET_ALL
        else:
            return Fore.RED + '{:.5f}'.format(n) + Style.RESET_ALL

    if unit == 'rad':
        n_deg = n*180/math.pi
        if n_deg < 0.1:
            return Fore.LIGHTGREEN_EX + '{:.5f}'.format(n) + Style.RESET_ALL
        if n_deg < 0.5:
            return Fore.GREEN + '{:.5f}'.format(n) + Style.RESET_ALL
        elif n_deg < 1:
            return Fore.YELLOW + '{:.5f}'.format(n) + Style.RESET_ALL
        elif n_deg < 3:
            return Fore.MAGENTA + '{:.5f}'.format(n) + Style.RESET_ALL
        else:
            return Fore.RED + '{:.5f}'.format(n) + Style.RESET_ALL


def addAveragesBottomRowToTable(table, header):

    # Compute averages and add a bottom row
    bottom_row = []  # Compute averages and add bottom row to table
    for col_idx, _ in enumerate(header):
        if col_idx == 0:
            bottom_row.append(Fore.BLUE + Style.BRIGHT + 'Averages' + Fore.BLACK + Style.RESET_ALL)
            continue

        total = 0
        count = 0
        for row in table.rows:
            cell_without_color_codes = removeColorsFromText(row[col_idx])

            try:
                value = float(cell_without_color_codes)
                total += float(value)
                count += 1
            except:
                pass

        if count > 0:
            value = '%.5f' % (total / count)
        else:
            value = '---'

        bottom_row.append(Fore.BLUE + value + Fore.BLACK + Style.RESET_ALL)

    table.add_row(bottom_row)
    return table


def printComparisonToGroundTruth(
        dataset, dataset_initial, dataset_ground_truth, selected_collection_key, output_folder, args):

    # --------------------------------------------------
    # Evaluate sensor poses
    # --------------------------------------------------
    for sensor_key, sensor in dataset["sensors"].items():
        header = ['Transform', 'Description', 'Et0 [m]', 'Et [m]', 'Rrot0 [rad]', 'Erot [rad]']
        table = PrettyTable(header)

        # Create a table_to_save to be output as a csv file (#977) 
        if args["save_file_results"]:
            header_table_to_save = ['Transform', 'Et [m]', 'Erot [rad]']
            table_to_save = PrettyTable(header_table_to_save)

        transform_key = generateKey(sensor["calibration_parent"], sensor["calibration_child"])
        row = [transform_key, Fore.BLUE + sensor_key + Style.RESET_ALL]

        transform_calibrated = dataset['collections'][selected_collection_key]['transforms'][
            transform_key]
        transform_ground_truth = dataset_ground_truth['collections'][selected_collection_key][
            'transforms'][transform_key]
        transform_initial = dataset_initial['collections'][selected_collection_key]['transforms'][transform_key]

        translation_error_1, rotation_error_1 = compareAtomTransforms(
            transform_initial, transform_ground_truth)
        translation_error_2, rotation_error_2 = compareAtomTransforms(
            transform_calibrated, transform_ground_truth)

        row.append(getNumberQualifier(translation_error_1))
        row.append(getNumberQualifier(translation_error_2))
        row.append(getNumberQualifier(rotation_error_1, unit='rad'))
        row.append(getNumberQualifier(rotation_error_2, unit='rad'))

        table.add_row(row)

        if args["save_file_results"]:
            row_table_to_save = [transform_key, round(translation_error_2,6), round(rotation_error_2, 6)]
            table_to_save.add_row(row_table_to_save)


    # TODO Evaluate intrinsics

    # --------------------------------------------------
    # Evaluate additional transforms
    # --------------------------------------------------
    if dataset['calibration_config']['additional_tfs'] is not None:
        for additional_tf_key, additional_tf in dataset['calibration_config']['additional_tfs'].items():

            transform_key = generateKey(additional_tf["parent_link"], additional_tf["child_link"])
            row = [transform_key, Fore.LIGHTCYAN_EX + additional_tf_key + Style.RESET_ALL]



            transform_calibrated = dataset['collections'][selected_collection_key]['transforms'][
                transform_key]
            transform_ground_truth = dataset_ground_truth['collections'][selected_collection_key][
                'transforms'][transform_key]
            transform_initial = dataset_initial['collections'][selected_collection_key][
                'transforms'][transform_key]

            translation_error_1, rotation_error_1 = compareAtomTransforms(
                transform_initial, transform_ground_truth)
            translation_error_2, rotation_error_2 = compareAtomTransforms(
                transform_calibrated, transform_ground_truth)

            row.append(getNumberQualifier(translation_error_1))
            row.append(getNumberQualifier(translation_error_2))
            row.append(getNumberQualifier(rotation_error_1, unit='rad'))
            row.append(getNumberQualifier(rotation_error_2, unit='rad'))
            table.add_row(row)
        
        if args["save_file_results"]:
            row_table_to_save = [transform_key, round(translation_error_2,6), round(rotation_error_2, 6)]
            table_to_save.add_row(row_table_to_save)
    
    # --------------------------------------------------
    # Evaluate pattern transforms
    # --------------------------------------------------
    for pattern_key, pattern in dataset['calibration_config']['calibration_patterns'].items():

        print('pattern ' + pattern_key)
        print(pattern)

        transform_key = generateKey(pattern["parent_link"], pattern["link"])
        row = [transform_key, Fore.LIGHTCYAN_EX + pattern_key + Style.RESET_ALL]

        transform_calibrated = dataset['collections'][selected_collection_key]['transforms'][
            transform_key]

        if transform_key not in dataset_ground_truth['collections'][selected_collection_key][
                'transforms']:
            atomWarn('Cannot print comparison to ground truth for pattern ' +
                     pattern_key + ' because there is no ground truth data.')
            continue

        transform_ground_truth = dataset_ground_truth['collections'][selected_collection_key][
            'transforms'][transform_key]
        transform_initial = dataset_initial['collections'][selected_collection_key]['transforms'][transform_key]

        translation_error_1, rotation_error_1 = compareAtomTransforms(
            transform_initial, transform_ground_truth)
        translation_error_2, rotation_error_2 = compareAtomTransforms(
            transform_calibrated, transform_ground_truth)

        row.append(getNumberQualifier(translation_error_1))
        row.append(getNumberQualifier(translation_error_2))
        row.append(getNumberQualifier(rotation_error_1, unit='rad'))
        row.append(getNumberQualifier(rotation_error_2, unit='rad'))
        table.add_row(row)

        if args["save_file_results"]:
            row_table_to_save = [transform_key, round(translation_error_2,6), round(rotation_error_2, 6)]
            table_to_save.add_row(row_table_to_save)

    # Add bottom row with averages
    table = addAveragesBottomRowToTable(table, header)

    print(Style.BRIGHT + '\nTransforms Calibration' + Style.RESET_ALL)
    print(
        'Et: average translation error (Et0 - initial) [m],  Erot: average rotation error (Erot0 - initial) [rad]')
    print('Translation errors: ' + Fore.LIGHTGREEN_EX + '< 1 mm' + Fore.BLACK + ' | ' + Fore.GREEN +
          '< 1 cm' + Fore.BLACK + ' | ' + Fore.YELLOW + '< 5 cm' + Fore.BLACK + ' | ' + Fore.RED +
          '>= 5 cm' + Style.RESET_ALL)
    print('Rotation errors: ' + Fore.LIGHTGREEN_EX + '< 0.1 deg' + Fore.BLACK + ' | ' + Fore.GREEN +
          '< 0.5 deg' + Fore.BLACK + ' | ' + Fore.YELLOW + '< 1 deg' + Fore.BLACK + ' | ' + Fore.MAGENTA +
          '< 3 deg' + Fore.BLACK + ' | ' + Fore.RED + '>= 3 deg' + Style.RESET_ALL)

    print(table)
    filename = output_folder + '/comparison_to_ground_truth_transforms.csv'
    print('Saving transforms ground truth comparison to ' + Fore.BLUE + filename + Style.RESET_ALL)
    with open(filename, 'w', newline='') as file:
        file.write(removeColorsFromText(table.get_csv_string()))

    # Save the results in a csv file (TFs in a separate file to the joint params)
    # save results in csv file
    if args['save_file_results']:
        if args['save_file_results_name'] is None:
            results_name = 'comparison_to_gt_tfs.csv'
        else:
            results_name = args['save_file_results_name']

        with open(results_name, 'w', newline='') as f_output:
            f_output.write(table_to_save.get_csv_string())

    # --------------------------------------------------
    # Evaluate joints
    # --------------------------------------------------

    if dataset['calibration_config']['joints'] is not None:
        header = ['Joint', 'Param', 'Error (ini)', 'Error (calib)']
        table = PrettyTable(header) 

        for joint_key, joint in dataset['calibration_config']['joints'].items():
            for param in joint['params_to_calibrate']:
                row = [joint_key, param]

                value_calibrated = dataset['collections'][selected_collection_key]['joints'][
                    joint_key][param]
                value_ground_truth = dataset_ground_truth['collections'][selected_collection_key][
                    'joints'][joint_key][param]
                value_initial = dataset_initial['collections'][selected_collection_key]['joints'][joint_key][param]

                error_initial = getNumberQualifier(
                    abs(value_ground_truth-value_initial), unit='rad')
                row.append(error_initial)

                error_calibrated = getNumberQualifier(
                    abs(value_ground_truth-value_calibrated), unit='rad')
                row.append(error_calibrated)

                table.add_row(row)

        # Add bottom row with averages
        table = addAveragesBottomRowToTable(table, header)

        print(Style.BRIGHT + '\nJoints Calibration' + Style.RESET_ALL)
        print(
            'Translation errors: ' + Fore.LIGHTGREEN_EX + '< 1 mm' + Fore.BLACK + ' | ' + Fore.GREEN +
            '< 1 cm' + Fore.BLACK + ' | ' + Fore.YELLOW + '< 5 cm' + Fore.BLACK + ' | ' + Fore.RED +
            '>= 5 cm' + Style.RESET_ALL)
        print(
            'Rotation errors: ' + Fore.LIGHTGREEN_EX + '< 0.1 deg' + Fore.BLACK + ' | ' + Fore.GREEN +
            '< 0.5 deg' + Fore.BLACK + ' | ' + Fore.YELLOW + '< 1 deg' + Fore.BLACK + ' | ' + Fore.MAGENTA
            + '< 3 deg' + Fore.BLACK + ' | ' + Fore.RED + '>= 3 deg' + Style.RESET_ALL)

        print(table)

        filename = output_folder + '/comparison_to_ground_truth_joints.csv'
        print('Saving joints ground truth comparison to ' + Fore.BLUE + filename + Style.RESET_ALL)
        with open(filename, 'w', newline='') as file:
            file.write(removeColorsFromText(table.get_csv_string()))

    

def raise_timeout_error(signum, frame):
    raise subprocess.TimeoutExpired(None, 1)


def input_with_timeout(prompt, timeout):
    # set signal handler
    signal(SIGALRM, raise_timeout_error)
    setitimer(ITIMER_REAL, timeout)

    try:
        k = readchar.readkey()
        print('received key')
        return k
    except:
        # print('Timeout')
        pass
    finally:
        # print('finally')
        # signal.alarm(0)  # cancel alarm
        setitimer(ITIMER_REAL, 0)


def waitForKeyPress2(function=None, timeout=5, message='Waiting ... '):
    message = message + '\npress ' + Fore.BLUE + Style.BRIGHT + '"c"' + Style.RESET_ALL + \
        ' to continue or ' + Fore.BLUE + Style.BRIGHT + '"q"' + Style.RESET_ALL + ' to abort.'

    while True:
        if not function is None:
            print('Calling function')
            function()

        print(message)
        key = input_with_timeout('', 1)
        print('key: ' + str(key))
        if key == 'c':
            print('\nyou pressed c ... continuing.')
            break
        elif key == 'q':
            print('\nyou pressed q ... aborting.')
            exit(0)


def waitForKeyPress(function=None, timeout=5, message='Waiting ... '):
    message = message + '\npress ' + Fore.BLUE + Style.BRIGHT + '"c"' + Style.RESET_ALL + \
        ' to continue or ' + Fore.BLUE + Style.BRIGHT + '"q"' + Style.RESET_ALL + ' to abort.'

    while True:
        with keyboard.Events() as events:
            if not function is None:
                print('Calling function')
                function()

            print(message)
            event = events.get(timeout)
            if not event is None:
                if hasattr(event.key, 'char'):
                    if event.key.char == 'c':
                        print('\nyou pressed c ... continuing.')
                        break
                    elif event.key.char == 'q':
                        print('\nyou pressed q ... aborting.')
                        exit(0)


# Check https://stackoverflow.com/questions/52431265/how-to-use-a-lambda-as-parameter-in-python-argparse
def create_lambda_with_globals(s):
    return eval(s, globals())


def laser_scan_msg_to_xy(msg):
    data = message_converter.convert_ros_message_to_dictionary(msg)
    return laser_scan_data_to_xy(data)


def laser_scan_data_to_xy(data):
    ranges = data['ranges']
    angle_min = data['angle_min']
    angle_increment = data['angle_increment']

    x = []
    y = []
    for range_idx, r in enumerate(ranges):
        theta = angle_min + angle_increment * range_idx
        x.append(r * math.cos(theta))
        y.append(r * math.sin(theta))

    return x, y


def checkDirectoryExistence(directory, package_name, create_if_nonexistent=False):
    # full path to the package, including its name.
    package_path = rospkg.RosPack().get_path(package_name)

    if not os.path.exists(package_path + '/' + directory):
        if create_if_nonexistent:
            print(Fore.YELLOW + directory + Style.RESET_ALL +
                  ' directory does not exist. Created a new one.')
            execute('mkdir ' + package_path + '/' + directory)
            return True
        else:
            print(Fore.RED + directory + Style.RESET_ALL + ' directory does not exist.')
            return False


def rootMeanSquare(errors):
    """Computes mean square error of a list of error values.

    Args:
        errors (list of float): the list of errors
    """

    return math.sqrt(mean([e**2 for e in errors]))


def atomError(message):
    print(Fore.RED + '\nATOM Error: ' + Style.RESET_ALL + message)
    exit(0)


def atomWarn(message):
    print(Fore.YELLOW + 'ATOM Warn: ' + Style.RESET_ALL + message)


def atomPrintOK(message=''):
    print(message + Fore.GREEN + '[OK]' + Style.RESET_ALL)


def atomStartupPrint(message=''):
    print('_______________________________________________________')
    print("\n \
          █████╗ ████████╗ ██████╗ ███╗   ███╗ \n \
         ██╔══██╗╚══██╔══╝██╔═══██╗████╗ ████║ \n \
         ███████║   ██║   ██║   ██║██╔████╔██║ \n \
         ██╔══██║   ██║   ██║   ██║██║╚██╔╝██║ \n \
  __     ██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║    _  \n\
 / _|     ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝  | |    \n\
 | |_ _ __ __ _ _ __ ___   _____      _____  _ __| | __ \n\
 |  _| '__/ _` | '_ ` _ \ / _ \ \ /\ / / _ \| '__| |/ / \n\
 | | | | | (_| | | | | | |  __/\ V  V / (_) | |  |   <  \n\
 |_| |_|  \__,_|_| |_| |_|\___| \_/\_/ \___/|_|  |_|\_\ \n\
 " + Fore.BLUE + "https://github.com/lardemua/atom\n" + Style.RESET_ALL + '\n' + message)
    print('_______________________________________________________\n')


def verifyAnchoredSensor(anchored_sensor, sensors):
    # If anchored sensor exists, it must be one of the existing sensors
    print('Checking if anchored sensor ' + Fore.BLUE + str(anchored_sensor) +
          Style.RESET_ALL + ' is valid ... ' + Style.RESET_ALL, end='')

    if anchored_sensor is not None and not anchored_sensor in list(sensors.keys()):
        atomError('Anchored sensor ' + Fore.BLUE + anchored_sensor + Style.RESET_ALL +
                  ' must be one of the configured sensors or an empty field.')


def saveFileResults(train_json, test_json, results_name, table_to_save):
    dataset_name = train_json.split('/')[-1].split('.')[0]
    last_slash_index = test_json.rfind('/')
    # Remove everything after the last '/'
    folder_name = test_json[:last_slash_index]
    if not os.path.exists(folder_name + '/results'):
        os.makedirs(folder_name + '/results')
    with open(folder_name + '/results/' + dataset_name + '_' + results_name, 'w', newline='') as f_output:
        f_output.write(table_to_save.get_csv_string())


def verifyFixedPattern(dataset, pattern_key):
    print(
        f'Checking if calibration pattern {Fore.BLUE}{pattern_key}{Style.RESET_ALL} is fixed ...',
        end='')

    return dataset['calibration_config']['calibration_patterns'][pattern_key]['fixed']


def getJointParentChild(joint_key, description):
    # From the xacro, get the parent and child of this joint
    parent = None
    child = None
    for joint in description.joints:
        if joint.name == joint_key:
            parent = joint.parent
            child = joint.child

    return joint.parent, joint.child


def saveCommandLineArgsYml(args, output_file):
    with open(output_file, 'w') as file:
        yaml.dump(args, file, sort_keys=False)


def createLambdaExpressionsForArgs(args, keys_to_check=['sensor_selection_function',
                                                        'collection_selection_function',
                                                        'joint_selection_function',
                                                        'joint_parameter_selection_function',
                                                        'pattern_selection_function',
                                                        'additional_tf_selection_function']):

    args_with_lambdas = deepcopy(args)
    for arg_key in keys_to_check:

        if arg_key not in args_with_lambdas:
            continue

        if args_with_lambdas[arg_key] is None:  # nothing to do
            continue

        print('Creating lambda expression for arg ' + Fore.BLUE + arg_key + Style.RESET_ALL)
        args_with_lambdas[arg_key] = eval(args_with_lambdas[arg_key], globals())

    return args_with_lambdas
