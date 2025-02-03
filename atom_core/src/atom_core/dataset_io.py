import copy

# Standard imports
import json
import os
from os.path import exists

import numpy as np

# Opencv imports
import cv2
from atom_core.joint_models import getTransformationFromJoint
from atom_core.utilities import atomError, atomWarn

# Ros imports
import rospy
import tf
import atom_core.pypcd as pypcd
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, sensor_msgs

# Atom imports
from cv_bridge import CvBridge
from colorama import Fore, Style
from rospy_message_converter import message_converter
from std_msgs.msg import Header
from atom_core.config_io import uriReader, mutually_inclusive_conditions
from atom_core.naming import generateName, generateKey, generateCollectionKey
from atom_calibration.collect.label_messages import (
    convertDepthImage32FC1to16UC1, convertDepthImage16UC1to32FC1, numpyFromPointCloudMsg)


def printImageInfo(image, text=None):
    if not text is None:
        print(text +
              '\n\tshape = ' + str(image.shape) +
              '\n\tdtype = ' + str(image.dtype) +
              '\n\tmax value = ' + str(np.nanmax(image)) +
              '\n\tmin value = ' + str(np.nanmin(image)))


def loadResultsJSON(json_file, collection_selection_function=None):

    dataset = loadJSONFile(json_file)
    dataset_folder = os.path.dirname(json_file)
    bridge = CvBridge()

    # Load images from files into memory. Images in the json file are stored in separate png files and in their place
    # a field "data_file" is saved with the path to the file. We must load the images from the disk.
    # Do the same for point clouds saved in pcd files
    skipped_loading = []
    for collection_key, collection in dataset['collections'].items():

        # Check if collection is listed to be ignored by csf and do not load image and point cloud if it is
        if not collection_selection_function is None:
            if not collection_selection_function(collection_key):  # use the lambda expression csf
                skipped_loading.append(collection_key)
                continue

        for sensor_key, sensor in dataset['sensors'].items():

            if not (sensor['modality'] == 'rgb' or sensor['modality'] == 'lidar3d' or
                    sensor['modality'] == 'lidar2d' or sensor['modality'] == 'depth'):
                continue  # only process images or point clouds

            # Check if we really need to load the file.
            if 'data' in collection['data'][sensor_key]:
                load_file = False
            elif 'data_file' in collection['data'][sensor_key]:
                filename = dataset_folder + '/' + collection['data'][sensor_key]['data_file']
                if os.path.isfile(filename):
                    load_file = True
                else:
                    raise ValueError(
                        'Datafile points to ' + collection['data'][sensor_key]['data_file'] +
                        ' but file ' + filename + ' does not exist.')
            else:
                raise ValueError('Dataset does not contain data nor data_file folders.')

            if load_file and (sensor['modality'] == 'rgb'):  # Load image.

                filename = os.path.dirname(
                    json_file) + '/' + collection['data'][sensor_key]['data_file']
                cv_image = cv2.imread(filename)  # Load image from file
                dict_image = getDictionaryFromCvImage(cv_image)  # from opencv image to dictionary

                # Check if loaded image has the same properties as the dataset in collection['data'][sensor_key]
                assert collection['data'][sensor_key]['height'] == dict_image['height'], 'Image height must be the same'
                assert collection['data'][sensor_key]['width'] == dict_image['width'], 'Image width must be the same'

                # set data field of collection
                collection['data'][sensor_key]['data'] = dict_image['data']
                collection['data'][sensor_key]['encoding'] = dict_image['encoding']
                collection['data'][sensor_key]['step'] = dict_image['step']
                # Previous code, did not preserve frame_id and other properties
                # collection['data'][sensor_key].update(getDictionaryFromCvImage(cv_image))

            elif load_file and sensor['modality'] == 'depth':
                filename = os.path.dirname(
                    json_file) + '/' + collection['data'][sensor_key]['data_file']

                # print(collection['data'][sensor_key]['header']['frame_id'])
                cv_image_int16_tenths_of_millimeters = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                cv_image_float32_meters = convertDepthImage16UC1to32FC1(
                    cv_image_int16_tenths_of_millimeters, scale=10000.0)
                # collection['data'][sensor_key]['encoding']='32FC1'

                # cv2.imshow("cv_image_float32_meters", cv_image_int16_tenths_of_millimeters)
                # cv2.waitKey(0)
                # imageShowUInt16OrFloat32OrBool(cv_image_float32_meters, "float32_load_file")
                # cv2.waitKey(5)

                # printImageInfo(cv_image_int16_tenths_of_millimeters, text='cv_image_int16_tenths_of_millimeters')
                # printImageInfo(cv_image_float32_meters, text='cv_image_float32_meters')

                # collection['data'][sensor_key].update(getDictionaryFromDepthImage(cv_image_float32_meters))

                dict = getDictionaryFromDepthImage(cv_image_float32_meters)

                collection['data'][sensor_key]['data'] = dict['data']
                collection['data'][sensor_key]['encoding'] = dict['encoding']
                collection['data'][sensor_key]['step'] = dict['step']
                # del dict['data']
                # del collection['data'][sensor_key]['data']
                # print(dict)
                # print(collection['data'][sensor_key])
                # exit(0)
                # msg_33 = message_converter.convert_dictionary_to_ros_message('sensor_msgs/Image', dict)
                # image_33=bridge.imgmsg_to_cv2(msg_33, desired_encoding='passthrough')
                # imageShowUInt16OrFloat32OrBool(image_33, "load_file_dic")
                # cv2.waitKey(5)

                # TODO eliminate data_file
                # TODO Why this is not needed for rgb? Should be done as well

                # print(collection['data'][sensor_key]['header']['frame_id'])
                # print(collection['data'][sensor_key].keys())
                # TODO verify if values in the dataset or ok
                # exit(0)

            # Load point cloud.
            elif load_file and (sensor['modality'] == 'lidar3d' or sensor['modality'] == 'lidar2d'):
                filename = os.path.dirname(
                    json_file) + '/' + collection['data'][sensor_key]['data_file']
                frame_id = str(collection['data'][sensor_key]['header']['frame_id'])

                # setup header for point cloud from existing dictionary data
                header = Header()
                header.frame_id = frame_id
                time = rospy.Time()
                time.secs = collection['data'][sensor_key]['header']['stamp']['secs']
                time.nsecs = collection['data'][sensor_key]['header']['stamp']['nsecs']
                header.stamp = time
                header.seq = collection['data'][sensor_key]['header']['seq']

                # read point cloud from dist
                msg = read_pcd(filename, cloud_header=header)

                # convert to dictionary
                collection['data'][sensor_key].update(
                    message_converter.convert_ros_message_to_dictionary(msg))

    if skipped_loading:  # list is not empty
        print('Skipped loading images and point clouds for collections: ' + str(skipped_loading) + '.')

    return dataset, json_file


def saveAtomDataset(filename, dataset_in, save_data_files=True, freeze_dataset=False):
    if freeze_dataset:  # to make sure our changes only affect the dictionary to save
        dataset = copy.deepcopy(dataset_in)
    else:
        dataset = dataset_in

    if save_data_files:
        # Process the dataset to remove data from the data fields and, if needed, write the files.
        for collection_key, collection in dataset['collections'].items():
            for sensor_key, sensor in dataset['sensors'].items():
                # print('Saving  collection ' + collection_key + ' sensor ' + sensor_key)
                createDataFile(dataset, collection_key, sensor,
                               sensor_key, os.path.dirname(filename))

        # Do the same for additional data topics ...
        for description, sensor in dataset['additional_sensor_data'].items():
            createDataFile(dataset_in, collection_key, sensor, description,
                           os.path.dirname(filename), 'additional_data')

    createJSONFile(filename, dataset)  # write dictionary to json


def createDataFile(dataset, collection_key, sensor, sensor_key, output_folder, data_type='data'):
    if not (sensor['modality'] == 'rgb' or sensor['modality'] == 'lidar3d' or
            sensor['modality'] == 'lidar2d' or sensor['modality'] == 'depth'):
        return

    # Check if data_file has to be created based on the existence of the field 'data_file' and the file itself.
    if 'data_file' in dataset['collections'][collection_key][data_type][sensor_key]:
        filename = output_folder + '/' + \
            dataset['collections'][collection_key][data_type][sensor_key]['data_file']
        if os.path.isfile(filename):
            create_data_file = False
            if 'data' in dataset['collections'][collection_key][data_type][sensor_key]:  # remove the data field
                del dataset['collections'][collection_key][data_type][sensor_key]['data']

        else:
            create_data_file = True
    else:
        create_data_file = True

    if create_data_file:
        # print('Collection ' + str(collection_key) + '. Creating data file for sensor ' + str(sensor_key)
        #   + ' msg type ' + sensor['msg_type'])
        pass

    if create_data_file and sensor['modality'] == 'rgb':  # save image.
        # Save image to disk if it does not exist
        filename = output_folder + '/' + sensor['_name'] + '_' + str(collection_key) + '.jpg'
        if not os.path.isfile(filename):  # Write rgb image file
            cv_image = getCvImageFromDictionary(
                dataset['collections'][collection_key][data_type][sensor_key])

            # flip color channels if needed
            if dataset['collections'][collection_key][data_type][sensor_key]['encoding'] == 'rgb8':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                dataset['collections'][collection_key][data_type][sensor_key]['encoding'] = 'bgr8'

            if not exists(filename):
                # print('Starting to save file ' + filename + '.')
                cv2.imwrite(filename, cv_image)
                print('Saved file ' + filename + '.')
            else:
                print('File ' + filename + ' already exists, skipping save ...')

        # Add data_file field, and remove data field
        filename_relative = sensor['_name'] + '_' + str(collection_key) + '.jpg'
        # add data_file
        dataset['collections'][collection_key][data_type][sensor_key]['data_file'] = filename_relative
        # Delete data field from dictionary
        if 'data' in dataset['collections'][collection_key][data_type][sensor_key]:
            del dataset['collections'][collection_key][data_type][sensor_key]['data']

    elif create_data_file and sensor['modality'] == 'lidar3d':  # save point cloud
        # sensor['modality'] == 'lidar2d':  # TODO Add for lidar 2D
        # Save file if it does not exist
        filename = output_folder + '/' + sensor['_name'] + '_' + str(collection_key) + '.pcd'
        if not os.path.isfile(filename):  # Write pointcloud to pcd file
            # TODO must remove this or True, just for debugging
            # from: dictionary -> ros_message -> PointCloud2() -> pcd file
            msg = getPointCloudMessageFromDictionary(
                dataset['collections'][collection_key][data_type][sensor_key])
            write_pcd(filename, msg)
            print('Saved file ' + filename + '.')

        # Add data_file field, and remove data field
        filename_relative = sensor['_name'] + '_' + str(collection_key) + '.pcd'
        dataset['collections'][collection_key][data_type][sensor_key][
            'data_file'] = filename_relative  # add data_file field
        # Delete data field from dictionary
        if 'data' in dataset['collections'][collection_key][data_type][sensor_key]:
            del dataset['collections'][collection_key][data_type][sensor_key]['data']

    elif create_data_file and sensor['modality'] == 'depth':
        # print('i dont know what im doing')
        # Save image to disk if it does not exist
        filename = output_folder + '/' + sensor['_name'] + '_' + str(collection_key) + '.png'
        # filename_np=output_folder + '/' + sensor['_name'] + '_' + str(collection_key) + '.npy'
        if not os.path.isfile(filename):  # Write pointcloud to pcd file
            cv_image = getCvImageFromDictionaryDepth(
                dataset['collections'][collection_key][data_type][sensor_key])
            # print("image from getimage")
            # print(cv_image.dtype)
            cv_image = convertDepthImage32FC1to16UC1(
                cv_image, scale=10000)  # Better to use tenths of milimeters
            # cv2.normalize(cv_image, cv_image, 0, 65535, cv2.NORM_MINMAX)
            cv2.imwrite(filename, cv_image)
            print('Saved file ' + filename + '.')

        # Add data_file field, and remove data field
        filename_relative = sensor['_name'] + '_' + str(collection_key) + '.png'
        dataset['collections'][collection_key][data_type][sensor_key][
            'data_file'] = filename_relative  # add data_file field
        # Delete data field from dictionary
        if 'data' in dataset['collections'][collection_key][data_type][sensor_key]:
            del dataset['collections'][collection_key][data_type][sensor_key]['data']


def getDictionaryFromCvImage(cv_image):
    """
    Creates a dictionary from the opencv image, going from cvimage -> ros_message -> dictionary.
    :param cv_image:  the image in opencv format.
    :return: A dictionary converted from the ros message.
    """
    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    return message_converter.convert_ros_message_to_dictionary(msg)


def getDictionaryFromDepthImage(cv_image):
    """
    Creates a dictionary from the opencv image, going from cvimage -> ros_message -> dictionary.
    :param cv_image:  the image in opencv format.
    :return: A dictionary converted from the ros message.
    """

    # imageShowUInt16OrFloat32OrBool(cv_image, "float32_getdictionary_in")
    # cv2.waitKey(5)

    bridge = CvBridge()
    msg = bridge.cv2_to_imgmsg(cv_image, "passthrough")

    # d=message_converter.convert_ros_message_to_dictionary(msg)
    # msg=message_converter.convert_dictionary_to_ros_message('sensor_msgs/Image',d)
    # image=bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # imageShowUInt16OrFloat32OrBool(image, "float32_getdictionary_out")
    # cv2.waitKey(5)

    return message_converter.convert_ros_message_to_dictionary(msg)


def getCvImageFromDictionary(dictionary_in, safe=False):
    """
    Converts a dictionary (read from a json file) into an opencv image.
    To do so it goes from dictionary -> ros_message -> cv_image
    :param dictionary_in: the dictionary read from the json file.
    :return: an opencv image.
    """
    if safe:
        d = copy.deepcopy(dictionary_in)  # to make sure we don't touch the dictionary
    else:
        d = dictionary_in

    if 'data_file' in d:  # Delete data field from dictionary
        del d['data_file']  # will disrupt the dictionary to ros message

    msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/Image', d)
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(msg)


def getCvImageFromDictionaryDepth(dictionary_in, safe=False, scale=1000.0):
    """
    Converts a dictionary (read from a json file) into an opencv image.
    To do so it goes from dictionary -> ros_message -> cv_image
    :param dictionary_in: the dictionary read from the json file.
    :return: an opencv image.
    """

    _, image = getMsgAndCvImageFromDictionaryDepth(dictionary_in, safe=safe, scale=scale)
    return image


def getMsgAndCvImageFromDictionaryDepth(dictionary_in, safe=False, scale=1000.0):
    """
    Converts a dictionary (read from a json file) into an opencv image and a ros message.
    To do so it goes from dictionary -> ros_message -> cv_image
    :param dictionary_in: the dictionary read from the json file.
    :return: an opencv image.
    """
    if safe:
        d = copy.deepcopy(dictionary_in)  # to make sure we don't touch the dictionary
    else:
        d = dictionary_in

    if 'data_file' in d:  # Delete data field from dictionary
        del d['data_file']  # will disrupt the dictionary to ros message

    msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/Image', d)
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    if image.dtype == np.uint16:
        image = convertDepthImage16UC1to32FC1(image, scale=scale)
    return msg, image


def getPointCloudMessageFromDictionary(dictionary_in):
    """
    Converts dictionary to PointCloud2 message.
    :param dictionary_in: dictionary.
    :return: a ros Pointcloud2 message.
    """
    d = copy.deepcopy(dictionary_in)  # to make sure we don't touch the dictionary

    if 'data_file' in d:  # Delete data field from dictionary
        del d['data_file']  # will disrupt the dictionary to ros message

    msg = message_converter.convert_dictionary_to_ros_message('sensor_msgs/PointCloud2', d)
    return msg


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# json.dumps(data, cls=NpEncoder)

def createJSONFile(output_file, data):
    """
    Creates the json file containing the results data.
    :param output_file: output file.
    :param data: data in dict format to save.
    """
    D = copy.deepcopy(data)
    walk(D)

    f = open(output_file, 'w')
    # to get only four decimal places on the json file
    json.encoder.FLOAT_REPR = lambda f: ("%.6f" % f)
    # print >> f, json.dumps(D, indent=2, sort_keys=True)
    f.write(json.dumps(D, indent=2, sort_keys=True, cls=NpEncoder))
    f.close()
    print("Saved json output file to " + str(output_file) + ".")


def loadJSONFile(json_file):
    """
    Loads the json file containing the dataset, without the data.
    :param json_file: json file to load.
    """

    json_file, _, _ = uriReader(json_file)

    f = open(json_file, 'r')
    dataset = json.load(f)

    f.close()  # Close the file to prevent memory leaks

    return dataset


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def walk(node):
    for key, item in node.items():
        if isinstance(item, dict):
            walk(item)
        else:
            if isinstance(item, np.ndarray) and key == 'data':  # to avoid saving images in the json
                del node[key]

            elif isinstance(item, np.ndarray):
                node[key] = item.tolist()
            pass


def write_pcd(filename, pointcloud, mode='binary'):
    """
    This is meant to replace the old write_pcd from Andre which broke when migrating to python3.
    :param filename:
    :param cloud_header:
    :return:
    """

    print('Reading point cloud from ' + Fore.BLUE + filename + Style.RESET_ALL)

    # Convert to flattened (height=1) before saving to pcd, because pypcd cannot save non flattened pointclouds to pcd.
    # https://github.com/lardemua/atom/issues/520
    pc_np = numpyFromPointCloudMsg(pointcloud)
    points = []  # Build a list of points.
    for idx in range(0, pc_np.shape[0]):
        points.append([pc_np[idx, 0], pc_np[idx, 1], pc_np[idx, 2]])

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]
    flattened_point_cloud = pc2.create_cloud(pointcloud.header, fields, points)

    pc = pypcd.PointCloud.from_msg(flattened_point_cloud)
    pc.save_pcd(filename, compression=mode)


def read_pcd(filename, cloud_header=None, verbose=False):
    """
    This is meant to replace the old read_pcd from Andre which broke when migrating to python3.
    :param filename:
    :param cloud_header:
    :return:
    """

    if not os.path.isfile(filename):
        raise Exception("[read_pcd] File " + filename + " does not exist.")

    if verbose:
        print('Reading point cloud from ' + Fore.BLUE + filename + Style.RESET_ALL)
    pc = pypcd.PointCloud.from_path(filename)

    cloud = pc.to_msg()
    if cloud_header is not None:
        # cloud.header = header
        cloud.header = cloud_header
        # print('This is it, header is ' + str(cloud_header))
    else:
        cloud.header.frame_id = "/pcd_cloud"

    return cloud


def genCollectionPrefix(collection_key, string):
    """ Standardized form of deriving a name with a collection related prefix. """
    return generateName(string, prefix='c' + str(collection_key), separator='_')
    # return 'c' + str(collection_key) + '_' + str(string)


def checkIfAtLeastOneLabeledCollectionPerSensor(dataset):
    """
    Verifies if there is at least one collection for each sensor in which the sensor labeled the pattern.
    Without this the calibration cannot carry on.
    :param dataset: the dataset
    """

    for pattern_key, pattern in dataset['calibration_config']['calibration_patterns'].items():
        for sensor_key in dataset['sensors']:
            one_detection = False
            for collection_key in dataset['collections'].keys():
                if dataset['collections'][collection_key]['labels'][pattern_key][sensor_key]['detected']:
                    one_detection = True

            if one_detection is False:
                raise ValueError('Sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL +
                                 ' does not have a single collection in which the pattern ' +
                                 pattern_key + ' is labeled.' + Fore.RED +
                                 ' Cannot calibrate this sensor.' + Style.RESET_ALL +
                                 ' Add more collections or remove this sensor.')


def filterSensorsFromDataset(dataset, args):
    """
    Filters some sensors from the dataset, using a couple of arguments in arg
    :param dataset:
    :param args: Makes use of 'sensor_selection_function'
    """

    if 'only_anchored_sensor' in args.keys():
        if args['only_anchored_sensor'] == True:
            if dataset['calibration_config']['anchored_sensor'] is None:
                raise ValueError(
                    'Option only_anchored_sensor selected but there is no anchored sensor.')

            deleted = []
            for sensor_key in dataset['sensors']:
                if not sensor_key == dataset['calibration_config']['anchored_sensor']:
                    deleted.append(sensor_key)

            for sensor_key in deleted:
                del dataset['sensors'][sensor_key]

            print("Deleted sensors (option only_anchored_sensor): " + str(deleted))

    if 'sensor_selection_function' in args:
        if not args['sensor_selection_function'] is None:
            deleted = []
            for sensor_key in dataset['sensors']:
                # use the lambda expression ssf
                if not args['sensor_selection_function'](sensor_key):
                    deleted.append(sensor_key)

            for sensor_key in deleted:
                del dataset['sensors'][sensor_key]

            print("Deleted sensors: " + str(deleted))

    if not dataset['sensors'].keys():
        raise ValueError(
            'No sensors were selected. Cannot optimize without sensors. Please revise your '
            'dataset and your sensor selection function.')

    return dataset


def filterCollectionsFromDataset(dataset, args):
    """
    Filters some collections from the dataset, using a couple of arguments in arg
    :param dataset:
    :param args: Makes use of 'collection_selection_function', 'use_incomplete_collections' and
                'remove_partial_detections'
    """

    if not args['collection_selection_function'] is None:
        deleted = []
        for collection_key in dataset['collections'].keys():
            # use the lambda expression csf
            if not args['collection_selection_function'](collection_key):
                deleted.append(collection_key)

        for collection_key in deleted:
            del dataset['collections'][collection_key]

        if deleted:  # list is not empty
            print('Deleted collections: ' + str(deleted) + ' because of the -csf flag.')

    if not args['use_incomplete_collections']:
        deleted = []
        # Deleting collections where the pattern is not found by all sensors:
        for collection_key, collection in dataset['collections'].items():
            for pattern_key, pattern in collection['labels'].items():
                for sensor_label_key, sensor_label in pattern.items():
                    # one pattern not detected is enough for this to be an incomplete collection
                    if not sensor_label['detected']:
                        deleted.append(collection_key)
                        break

        # Using multiple patterns we can now discard a collection more than once. Therefore we compute the unique set of collections to be deleted
        deleted = list(set(deleted))

        for collection_key in deleted:
            del dataset['collections'][collection_key]

        if deleted:  # list is not empty
            print('Deleted collections: ' + str(deleted) +
                  ' because these are incomplete. If you want to use them set the ' + Fore.BLUE +
                  'use_incomplete_collections' + Style.RESET_ALL + ' flag.')

    if args['remove_partial_detections']:
        for pattern_key, pattern in dataset['calibration_config']['calibration_patterns'].items():
            number_of_corners = int(pattern['dimension']['x']) * int(pattern['dimension']['y'])
            # Deleting labels in which not all corners are found:
            for collection_key, collection in dataset['collections'].items():
                for sensor_key, sensor_label in dataset['sensors'].items():
                    if sensor_label['modality'] == 'rgb' and collection['labels'][pattern_key][
                            sensor_key]['detected']:
                        if not len(collection['labels'][pattern_key][sensor_key]['idxs']) == number_of_corners:
                            print(Fore.RED + 'Partial detection removed:' + Style.RESET_ALL + ' label from collection ' +
                                  collection_key + ' and pattern ' + Fore.BLUE + pattern_key + Style.RESET_ALL + ', sensor ' + sensor_key)
                            collection['labels'][pattern_key][sensor_key]['detected'] = False

    # It may occur that some collections do not have any detection in a camera sensor (because all detections were
    # partial and have been removed, or just because no detection existed). Since we need at lease one camera sensor
    # detection of the pattern in a collection in order initialize the parameters (check calibrate line 133),
    # we will remove collections which do not have at least one detection by a camera.
    flag_has_rgb_modality = False  # do this only if we have at least one camera in the sensor list.
    for sensor_key, sensor_label in dataset['sensors'].items():
        if sensor_label['modality'] == 'rgb':
            flag_has_rgb_modality = True
            break

    if flag_has_rgb_modality:
        deleted = []

        for collection_key, collection in dataset['collections'].items():
            flag_collection_has_at_least_one_rgb_detection = False
            for pattern_key, pattern in collection['labels'].items():
                for sensor_label_key, sensor_label in pattern.items():
                    modality = dataset['sensors'][sensor_label_key]['modality']
                    if modality == 'rgb' and sensor_label['detected']:
                        flag_collection_has_at_least_one_rgb_detection = True

            # delete collection without detection by cameras.
            if not flag_collection_has_at_least_one_rgb_detection:
                deleted.append(collection_key)

        for collection_key in deleted:
            del dataset['collections'][collection_key]

        if deleted:  # list is not empty
            print('Deleted collections: ' + str(deleted) +
                  ': at least one detection by a camera should be present.')

    if not dataset['collections'].keys():
        raise ValueError(
            'No collections were selected. Cannot optimize without collections. Please revise your '
            'dataset and your collection selection function.')

    print(
        'After filtering, will use ' + str(len(dataset['collections'].keys())) + ' collections: ' +
        str(list(dataset['collections'].keys())))

    return dataset


def filterPatternsFromDataset(dataset, args):
    """
    Filters some patterns from the dataset, using the pattern_selection_function
    :param dataset:
    :param args: Makes use of 'pattern_selection_function'
    """

    if 'pattern_selection_function' in args:
        if not args['pattern_selection_function'] is None:

            deleted = []
            for pattern_key in dataset['calibration_config']['calibration_patterns']:
                # use the lambda expression psf
                if not args['pattern_selection_function'](pattern_key):
                    deleted.append(pattern_key)

            for pattern_key in deleted:
                del dataset['calibration_config']['calibration_patterns'][pattern_key]
                for collection_key in dataset['collections']:
                    del dataset['collections'][collection_key]['labels'][pattern_key]

            print("Deleted patterns for calibration: " + str(deleted))

        if not dataset['calibration_config']['calibration_patterns']:
            atomError('There are no patterns in the dataset. Cannot continue.')

    return dataset


def filterJointParametersFromDataset(dataset, args):
    """
    Filters some joint parameters to be calibrated from the dataset, using the joint_parameter_selection_function
    :param dataset:
    :param args: Makes use of 'joint_parameter_selection_function'
    """

    if 'joint_parameter_selection_function' in args:
        if not args['joint_parameter_selection_function'] is None:
            if dataset['calibration_config']['joints'] is not None:
                for joint_key, joint in dataset['calibration_config']['joints'].items():

                    deleted = []
                    for param in joint['params_to_calibrate']:

                        # use the lambda expression jsf
                        if not args['joint_parameter_selection_function'](param):
                            deleted.append(param)

                    for param in deleted:
                        # print(dataset['calibration_config']['joints'][joint_key]['params_to_calibrate'])
                        dataset['calibration_config']['joints'][joint_key]['params_to_calibrate'].remove(
                            param)

                    print("Deleted parameters " + str(deleted) + ' from joint ' + joint_key)

    return dataset


def filterJointsFromDataset(dataset, args):
    """
    Filters some joints to be calibrated from the dataset, using the joint_selection_function
    :param dataset:
    :param args: Makes use of 'joint_selection_function'
    """

    if 'joint_selection_function' in args:
        if not args['joint_selection_function'] is None:
            deleted = []
            if dataset['calibration_config']['joints'] is not None:
                for joint_key in dataset['calibration_config']['joints']:
                    if not args['joint_selection_function'](joint_key):  # use the lambda expression jsf
                        deleted.append(joint_key)

            for joint_key in deleted:
                del dataset['calibration_config']['joints'][joint_key]
                for collection_key in dataset['collections']:
                    del dataset['collections'][collection_key]['joints'][joint_key]

            print("Deleted joints for calibration: " + str(deleted))

    return dataset


def filterAdditionalTfsFromDataset(dataset, args):
    """
    Filters some additional tfs to be calibrated from the dataset, using the additional_tfs_selection_function
    :param dataset:
    :param args: Makes use of 'additional_tf_selection_function'
    """

    if 'additional_tf_selection_function' in args:
        if not args['additional_tf_selection_function'] is None:
            deleted = []
            for additional_tf_key in dataset['calibration_config']['additional_tfs']:
                # use the lambda expression atsf
                if not args['additional_tf_selection_function'](additional_tf_key):
                    deleted.append(additional_tf_key)

            for additional_tf_key in deleted:
                del dataset['calibration_config']['additional_tfs'][additional_tf_key]

            print("Deleted additional_tfs for calibration: " + str(deleted))

    return dataset


def addBiasToJointParameters(dataset, args):
    """
    Adds noise
    :param dataset:
    :param args: Makes use of -jbn and -jbv
    """

    if dataset['calibration_config']['joints'] is None:  # nothing to do if no joints are being optimized
        print('No joints are being optimized')
        return

    if args['joint_bias_names'] is None and args['joint_bias_values'] is None and args['joint_bias_values'] is None:
        print('No bias to add to joints')
        return

    if np.any([args['joint_bias_names'] is None, args['joint_bias_values'] is None,
               args['joint_bias_values'] is None]):
        atomError('At least one of the arguments joint_bias_names, joint_bias_params or joint_bias_values are not given. These three must be used in tandem.')

    if len(args['joint_bias_names']) != len(args['joint_bias_params']) or \
            len(args['joint_bias_names']) != len(args['joint_bias_values']):
        atomError(
            'Args joint_bias_names, joint_bias_params and joint_bias_values must have the same size. Aborting.')

    for joint_name, joint_param, joint_bias in zip(
            args['joint_bias_names'],
            args['joint_bias_params'],
            args['joint_bias_values']):
        print('Adding bias ' + Fore.BLUE + str(joint_bias) + Style.RESET_ALL + ' to joint ' + Fore.GREEN +
              joint_name + Style.RESET_ALL + ' parameter ' + Fore.CYAN + joint_param + Style.RESET_ALL)
        for collection_key, collection in dataset['collections'].items():
            if joint_name not in collection['joints']:
                atomError(
                    'Cannot add noise to joint ' + Fore.BLUE + joint_name + Style.RESET_ALL +
                    ' parameter ' + Fore.BLUE + joint_param + Style.RESET_ALL + ' for collection ' +
                    collection_key)
                continue

            collection['joints'][joint_name][joint_param] = collection['joints'][joint_name][
                joint_param] + joint_bias

def addNoiseFromNoisyTFLinks(dataset,args,selected_collection_key):

    # Verify both arguments were provided
    # Unfortunately, mutually inclusive arguments are not built into argparse
    # https://github.com/python/cpython/issues/55797

    if not mutually_inclusive_conditions(args['noisy_tf_links'], args['noisy_tf_values']):
        return


    # Iterate through pairs of tf's and apply noise
    translation_tf_noise = args['noisy_tf_values'][0]
    rotation_tf_noise = args['noisy_tf_values'][1]

    for tf_pair in args['noisy_tf_links']:
        
        calibration_parent,calibration_child = tf_pair[0],tf_pair[1]
        addNoiseToTF(dataset,selected_collection_key,calibration_parent,calibration_child,translation_tf_noise,rotation_tf_noise)


def addNoiseToInitialGuess(dataset, args, selected_collection_key):
    """
    Adds noise
    :param dataset:
    :param args: Makes use of nig, i.e., the amount of noise to add to the initial guess atomic transformations to be
                 calibrated
    """
    # TODO create a issue to discuss if its ok to skip this function call when the noise is 0
    # if args['noisy_initial_guess'] == [0,0]:
    #     print("No noise added to transform's initial guess")
    #     return

    if args['sample_seed'] is not None:
        np.random.seed(args['sample_seed'])

    nig_trans = args['noisy_initial_guess'][0]
    nig_rot = args['noisy_initial_guess'][1]


    # Checking if tf to add noise is also defined in -ntfl, in which case the noise shouldn't be added here
    # Determining membership in sets is much faster than lists
    ntfl_tfs = set()
    if args['noisy_tf_links']:
        for tf_pair in args['noisy_tf_links']:
            ntfl_tfs.add(generateKey(tf_pair[0],tf_pair[1]))

    # add noise to additional tfs for simulation
    if dataset['calibration_config']['additional_tfs'] is not None:
        for _, additional_tf in dataset['calibration_config']['additional_tfs'].items():

            calibration_child = additional_tf['child_link']
            calibration_parent = additional_tf['parent_link']

            tf_to_add_noise = generateKey(calibration_parent,calibration_child)
            if tf_to_add_noise in ntfl_tfs:
                atomWarn(f'Not adding initial guess noise to {tf_to_add_noise} because its defined in -ntfl')
                continue

            addNoiseToTF(dataset, selected_collection_key, calibration_parent, calibration_child, nig_trans, nig_rot)

    # add noise to sensors tfs for simulation
    for sensor_key, sensor in dataset['sensors'].items():
        # if sensor_key == dataset['calibration_config']['anchored_sensor']:
        #     continue

        if sensor_key != dataset['calibration_config']['anchored_sensor']:
            calibration_child = sensor['calibration_child']
            calibration_parent = sensor['calibration_parent']

            tf_to_add_noise = generateKey(calibration_parent,calibration_child)
            if tf_to_add_noise in ntfl_tfs:
                atomWarn(f'Not adding initial guess noise to {tf_to_add_noise} because its defined in -ntfl')
                continue

            addNoiseToTF(dataset, selected_collection_key, calibration_parent, calibration_child, nig_trans, nig_rot)

# TODO make a basic function called by fixed and multiple

def computeNoise(initial_translation,initial_euler_angles,translation_noise_magnitude,rotation_noise_magnitude):
    '''
    Computes both the translation and rotation noise given a certain magnitude and returns the new noisy translation/rotation vectors
    '''
    # Translation

    v = np.random.uniform(-1.0, 1.0, 3)
    v = v / np.linalg.norm(v)
    translation_delta = v * translation_noise_magnitude

    new_translation = initial_translation + translation_delta
    # Rotation

    # Its necessary to redefine 'v' to ensure that the translation and rotation noise aren't always the same given the same magnitude
    v = np.random.uniform(-1.0, 1.0, 3)
    v = v / np.linalg.norm(v)
    rotation_delta = v * rotation_noise_magnitude

    new_euler_angles = initial_euler_angles + rotation_delta

    return new_translation,new_euler_angles


def addNoiseToTF(dataset, selected_collection_key, calibration_parent, calibration_child, noise_trans, noise_rot):

    print(Fore.RED + 'Transformation parent ' + calibration_parent + ' child ' + calibration_child + Style.RESET_ALL)
    transform_key = generateKey(calibration_parent, calibration_child, suffix='')

    atomWarn(f'Adding noise bigger than 1.0m/rad right now is bugged and might yield unexpected results. Check issue #929 for more information')

    # because of #900, and for retrocompatibility with old datasets, we will assume that if the transforms field does
    # not exist in the dataset, then the transformation is fixed
    if 'transforms' not in dataset or dataset['transforms'][transform_key]['type'] == 'fixed':

        # Get original transformation
        quat = dataset['collections'][selected_collection_key]['transforms'][transform_key]['quat']
        translation = dataset['collections'][selected_collection_key]['transforms'][transform_key]['trans']

        euler_angles = tf.transformations.euler_from_quaternion(quat)


        new_translation,new_euler_angles = computeNoise(translation,euler_angles,noise_trans,noise_rot)

        # Replace the original atomic transformations by the new noisy ones
        new_quat = tf.transformations.quaternion_from_euler(new_euler_angles[0], new_euler_angles[1], new_euler_angles[2])
        dataset['collections'][selected_collection_key]['transforms'][transform_key]['quat'] = new_quat
        dataset['collections'][selected_collection_key]['transforms'][transform_key]['trans'] = list(new_translation)

        # Copy randomized transform to all collections
        for collection_key, collection in dataset['collections'].items():
            dataset['collections'][collection_key]['transforms'][transform_key]['quat'] = \
                dataset['collections'][selected_collection_key]['transforms'][transform_key]['quat']
            dataset['collections'][collection_key]['transforms'][transform_key]['trans'] = \
                dataset['collections'][selected_collection_key]['transforms'][transform_key]['trans']

    elif dataset['transforms'][transform_key]['type'] == 'multiple':

        for collection_key, collection in dataset["collections"].items():

            quat = dataset['collections'][collection_key]['transforms'][transform_key]['quat']
            translation = dataset['collections'][collection_key]['transforms'][transform_key]['trans']

            euler_angles = tf.transformations.euler_from_quaternion(quat)

            new_translation,new_euler_angles = computeNoise(translation,euler_angles,noise_trans,noise_rot)

            new_quat = tf.transformations.quaternion_from_euler(new_euler_angles[0], new_euler_angles[1], new_euler_angles[2])

            # Replace the original atomic transformations by the new noisy ones
            dataset['collections'][collection_key]['transforms'][transform_key]['quat'] = new_quat
            dataset['collections'][collection_key]['transforms'][transform_key]['trans'] = list(new_translation)


def copyTFToDataset(calibration_parent, calibration_child, source_dataset, target_dataset):
    """
    Copy optimized transformations from a source dataset to a target dataset.

    It identifies the optimized transformation on the source dataset and copies it
    to all collections in the target dataset.

    :param calibration_parent: The calibration parent identifier.
    :param calibration_child: The calibration child identifier.
    :param source_dataset: The source dataset containing the optimized transformation.
    :param target_dataset: The target dataset where the optimized transformation will be copied.
    """

    # Generate a key for the transformation based on calibration parameters
    transform_name = generateKey(calibration_parent, calibration_child)

    # We can only optimize fixed transformations, so the optimized transform should be the same for all
    # collections. We select the first collection (selected_collection_key) and retrieve the optimized
    # transformation for that.
    selected_collection_key = list(source_dataset['collections'].keys())[0]
    optimized_transform = source_dataset['collections'][selected_collection_key]['transforms'][
        transform_name]

    # Iterate through all collections of the target dataset and replace the optimized transformation
    for collection_key, collection in target_dataset['collections'].items():
        collection['transforms'][transform_name]['quat'] = optimized_transform['quat']
        collection['transforms'][transform_name]['trans'] = optimized_transform['trans']


def getMixedDataset(train_dataset, test_dataset):
    """Creates a mixed dataset from the train and test datasets.

    This is used for evaluating, when we want the transformations between sensors (and also the intrinsics) estimated during calibration and stored in the train dataset, combined with previously unseen collections, which come from the test dataset.

    Args:
        train_dataset (dict): An ATOM dataset produced through calibration.
        test_dataset (dict): An ATOM dataset for testing.
    """

    # Make full of the test dataset. Then small bits from the train dataset are copied.
    mixed_dataset = copy.deepcopy(test_dataset)

    # Replace optimized transformations in the test dataset copying from the train dataset
    for _, sensor in train_dataset['sensors'].items():
        copyTFToDataset(
            sensor['calibration_parent'],
            sensor['calibration_child'],
            train_dataset, mixed_dataset)

    if train_dataset['calibration_config']['additional_tfs']:
        for _, additional_tf in mixed_dataset['calibration_config']['additional_tfs'].items():
            copyTFToDataset(
                additional_tf['parent_link'],
                additional_tf['child_link'],
                train_dataset, mixed_dataset)

    # Copy intrinsic parameters for cameras from train to mixed dataset.
    for train_sensor_key, train_sensor in train_dataset['sensors'].items():
        if train_sensor['msg_type'] == 'Image':
            mixed_dataset['sensors'][train_sensor_key]['camera_info']['D'] = train_sensor['camera_info']['D']
            mixed_dataset['sensors'][train_sensor_key]['camera_info']['K'] = train_sensor['camera_info']['K']
            mixed_dataset['sensors'][train_sensor_key]['camera_info']['P'] = train_sensor['camera_info']['P']
            mixed_dataset['sensors'][train_sensor_key]['camera_info']['R'] = train_sensor['camera_info']['R']

    # Because we know all joint parameters are static, we can use a single collection from the train dataset
    train_selected_collection_key = list(train_dataset['collections'].keys())[0]

    # Copy the joint parameters from the train to the mixed dataset
    if train_dataset['calibration_config']['joints'] is not None:

        for config_joint_key, config_joint in train_dataset['calibration_config']['joints'].items():

            for param_to_calibrate in config_joint['params_to_calibrate']:

                # Because we know all joint parameters are static throughout all collections, it means we can pick up the value for a single train_selected_collection_key and copy it to all collections in the mixed dataset.
                calibrated_value = train_dataset['collections'][train_selected_collection_key][
                    'joints'][config_joint_key][param_to_calibrate]

                # Now copy that value to all collections in the mixed dataset
                for collection_key, collection in mixed_dataset['collections'].items():
                    collection['joints'][config_joint_key][param_to_calibrate] = calibrated_value

        # Read all joints being optimized, and correct the corresponding transforms
        # print('Updating transforms from calibrated joints ...')
        for collection_key, collection in mixed_dataset['collections'].items():
            # print('Collection ' + collection_key)
            for joint_key, joint in collection['joints'].items():

                # Get the transformation from the joint configuration defined in the xacro, and the current joint value
                quat, trans = getTransformationFromJoint(joint)

                collection['transforms'][joint['transform_key']]['quat'] = quat
                collection['transforms'][joint['transform_key']]['trans'] = trans

    return mixed_dataset


def readAnnotationFile(json_file, sensor):
    annotations_file = os.path.dirname(json_file) + "/annotation_" + sensor + ".json"
    if os.path.exists(annotations_file) is False:
        print('Annotation file does not exist ' + Fore.RED + annotations_file + Style.RESET_ALL +
              '\nPlease annotate this sensor using: ' + Fore.BLUE +
              'rosrun atom_evaluation annotate_pattern_borders_in_rgb_or_depth --dataset ' +
              json_file + ' --rgb_sensor ' + sensor + Style.RESET_ALL)
        exit(0)

    print('Loading annotation file ' + Fore.BLUE + annotations_file + Style.RESET_ALL)
    annotations = json.load(open(annotations_file, 'r'))
    return annotations, annotations_file
