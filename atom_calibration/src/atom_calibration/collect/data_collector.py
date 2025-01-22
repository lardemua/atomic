# stdlib
from functools import partial
import json
import os
import time
import getpass
from datetime import datetime

import numpy as np
import rospkg
from atom_calibration.collect.patterns import estimatePatternPosesForCollection, initializePatternsDict
from atom_core.utilities import atomError, atomPrintOK
import tf2_ros
import yaml
import atom_core.config_io
import atom_core.dataset_io

# 3rd-party
from networkx.readwrite import json_graph
import rospy
import atom_msgs.srv
import tf
from matplotlib import cm
from cv_bridge import CvBridge
from colorama import Style, Fore
from interactive_markers.menu_handler import *
from rospy_message_converter import message_converter
from tf.listener import TransformListener
from sensor_msgs.msg import *
from urdf_parser_py.urdf import URDF

# local packages
from atom_core.naming import generateKey
from atom_core.ros_utils import printRosTime, getMaxTimeDelta, getAverageTime
from atom_core.config_io import loadConfig
from atom_core.system import execute, resolvePath
from atom_calibration.collect.configurable_tf_listener import ConfigurableTransformListener
from sensor_msgs.msg import JointState
from atom_msgs.msg import ImageWithRGBLabels, PointCloudWithLidar3DLabels, DepthImageWithDepthLabels


class DataCollector:

    def __init__(self, args, server, menu_handler):

        self.output_folder = resolvePath(args['output_folder'])

        # dataset path exists, abort
        if os.path.exists(self.output_folder) and not args['overwrite']:

            time.sleep(1.5)
            atomError('\n' + Fore.RED + 'Error: Dataset ' + self.output_folder +
                      ' exists.\nIf you want to replace it add a "--overwrite" flag.' + Style.RESET_ALL + '\n')

        # move existing path to a backup location
        elif os.path.exists(self.output_folder) and args['overwrite']:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            basename = os.path.basename(self.output_folder)
            backup_folder = '/tmp/' + basename + '_' + dt_string

            time.sleep(0.5)
            print('\n\nWarning: Dataset ' + Fore.YELLOW + self.output_folder + Style.RESET_ALL +
                  ' exists.\nMoving it to temporary folder: ' + Fore.YELLOW + backup_folder +
                  '\nThis will be deleted after a system reboot! If you want to keep it, copy the folder to some other location.' + Style.RESET_ALL + '\n\n')
            time.sleep(0.1)

            execute('mv ' + self.output_folder +
                    ' ' + backup_folder, verbose=True)
            # Create the folder empty
            os.makedirs(self.output_folder, exist_ok=False)

        elif not os.path.exists(self.output_folder):
            # Create the folder
            os.makedirs(self.output_folder, exist_ok=False)

        self.listener = TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.listener_2 = ConfigurableTransformListener(self.tf_buffer,
                                                        tf_topic='tf',
                                                        tf_static_topic='tf_static')

        self.sensors = {}
        self.transforms = {}
        self.server = server
        self.menu_handler = menu_handler
        self.data_stamp = 0
        self.timestamp = ""
        self.date = ""
        self.user = ""
        self.collections = {}
        self.additional_data = {}
        self.metadata = {}
        self.bridge = CvBridge()
        self.dataset_version = "3.0"  # included joint calibration
        self.joint_state_position_dict = {}

        # print(args['calibration_file'])
        self.config = loadConfig(args['calibration_file'])
        if self.config is None:
            sys.exit(1)  # loadJSON should tell you why.

        self.world_link = self.config['world_link']

        # Create a colormap so that we have one color per sensor
        self.cm_sensors = cm.Pastel2(np.linspace(
            0, 1, len(self.config['sensors'].keys())))

        # Reading the xacro
        self.dataset_name = self.output_folder.split('/')[-1]
        description_file, _, _ = atom_core.config_io.uriReader(
            self.config['description_file'])

        # Must first convert to urdf, if it is a xacro
        urdf_file = '/tmp/description.urdf'
        if os.path.exists(urdf_file):
            # print('Deleting temporary file ' + urdf_file)
            os.remove(urdf_file)

        print('Parsing description file ' + Fore.BLUE +
              description_file + Style.RESET_ALL)
        xacro_cmd = 'xacro ' + description_file + ' -o ' + urdf_file
        execute(xacro_cmd, verbose=True)  # create tmp urdf file

        if not os.path.exists(urdf_file):
            atomError('Could not parse description file ' + Fore.BLUE + description_file + Style.RESET_ALL + '\nYou must manually run command:\n' +
                      Fore.BLUE + xacro_cmd + Style.RESET_ALL + '\nand fix the problem before configuring your calibration package.')

        self.urdf_description = URDF.from_xml_file(
            urdf_file)  # read the urdf file

        # Setup joint_state message subscriber
        self.subscriber_joint_states = rospy.Subscriber(
            '/joint_states', JointState, self.callbackReceivedJointStateMsg, queue_size=1)

        # Configure patterns (compute corners positions, etc.)
        print('Initializing patterns ... ', end='')
        self.patterns_dict = initializePatternsDict(self.config)
        atomPrintOK()

        # load the transforms_graph.json
        rospack = rospkg.RosPack()
        # full path to the package, including its name.
        package_path = rospack.get_path(self.config['package_name'])
        transforms_graph_file = package_path + '/calibration/transforms_graph.json'

        if not os.path.exists(transforms_graph_file):  # file does not exist, abort
            atomError('Transforms graph file should exist at ' + Fore.BLUE +
                      transforms_graph_file + Style.RESET_ALL +
                      '\nReconfigure your package and try again.')

        f = open(transforms_graph_file, 'r')
        d = json.load(f)

        # Create a dictionary where the keys are the transform keys in the format
        # parent-child.
        for edge in d['links']:
            transform_key = generateKey(edge['source'], edge["target"])
            # remove keys source and target from edge before saving
            del edge['source']
            del edge['target']
            # only the edges of the graph are needed
            self.transforms[transform_key] = edge

        # Add sensors
        print(Fore.BLUE + 'Sensors:' + Style.RESET_ALL)
        print('Number of sensors: ' + str(len(self.config['sensors'])))

        # Build a dictionary containing information for each sensor
        label_data = {}
        for sensor_key, value in self.config['sensors'].items():

            print('Configuring sensor ' + sensor_key)
            # Create a dictionary that describes this sensor
            sensor_dict = {'_name': sensor_key, 'modality': value['modality'], 'parent': value['link'],
                           'calibration_parent': value['parent_link'],
                           'calibration_child': value['child_link']}

            # TODO replace by utils function
            print("Waiting for message " + value['topic_name'] + ' ...')
            msg = rospy.wait_for_message(value['topic_name'], rospy.AnyMsg)
            print('... received!')
            connection_header = msg._connection_header['type'].split('/')
            ros_pkg = connection_header[0] + '.msg'
            msg_type = connection_header[1]
            print('Topic ' + value['topic_name'] + ' has type ' + msg_type)
            sensor_dict['topic'] = value['topic_name']
            sensor_dict['msg_type'] = msg_type
            modality = value['modality']

            # If topic contains a message type then get a camera_info message to store along with the sensor data
            if modality == 'rgb' or modality == 'depth':  # if it is an image must get camera_info
                sensor_dict['camera_info_topic'] = os.path.dirname(
                    sensor_dict['topic']) + '/camera_info'
                from sensor_msgs.msg import CameraInfo
                print('Waiting for camera_info message on topic ' +
                      sensor_dict['camera_info_topic'] + ' ...')
                camera_info_msg = rospy.wait_for_message(
                    sensor_dict['camera_info_topic'], CameraInfo)
                print('... received!')
                from rospy_message_converter import message_converter
                sensor_dict['camera_info'] = message_converter.convert_ros_message_to_dictionary(
                    camera_info_msg)
                # sensor_dict['camera_info_msg'] = camera_info_msg
                # print(camera_info_msg)

            # Get the kinematic chain from world_link to this sensor's parent link
            now = rospy.Time()
            print('Waiting for transformation from ' + Fore.BLUE +
                  value['link'] + Style.RESET_ALL + ' to ' + Fore.BLUE + self.world_link + Style.RESET_ALL)
            self.listener.waitForTransform(
                value['link'], self.world_link, now, rospy.Duration(5))
            print('... received!')
            chain = self.listener.chain(
                value['link'], now, self.world_link, now, self.world_link)

            chain_list = []
            for parent, child in zip(chain[0::], chain[1::]):
                key = generateKey(parent, child)
                chain_list.append(
                    {'key': key, 'parent': parent, 'child': child})

            sensor_dict['chain'] = chain_list  # Add to sensor dictionary
            self.sensors[sensor_key] = sensor_dict

            label_data[sensor_key] = True
            if not args['skip_sensor_labeling'] is None:
                # use the lambda expression csf
                if args['skip_sensor_labeling'](sensor_key):
                    label_data[sensor_key] = False

            print('Configuration for sensor ' + Fore.BLUE +
                  sensor_key + Style.RESET_ALL + ' is complete.')

        # Defining metadata
        self.metadata = {"timestamp": str(time.time()), "date": time.ctime(time.time()), "user": getpass.getuser(),
                         'version': self.dataset_version, 'robot_name': self.urdf_description.name,
                         'dataset_name': self.dataset_name, 'package_name': self.config['package_name']}

        self.abstract_transforms = self.getAllAbstractTransforms()
        # print("abstract_transforms = " + str(self.abstract_transforms))

        # Add service to make the dataset json available
        self.service_get_dataset = rospy.Service('~get_dataset',
                                                 atom_msgs.srv.GetDataset,
                                                 self.callbackGetDataset)
        # Add service to save a new collection
        self.service_save_collection = rospy.Service('~save_collection',
                                                     atom_msgs.srv.SaveCollection,
                                                     self.callbackSaveCollection)

        # Add service to delete a new collection
        self.service_delete_collection = rospy.Service('~delete_collection',
                                                       atom_msgs.srv.DeleteCollection,
                                                       self.callbackDeleteCollection)

        # Create the label_msgs dictionary and fill it with the first messages
        # Also create the msg_buffer dict for continuous data collection
        # NOTE: IMUs don't make use of labelers, so the label_msg_type and the labels_topic are simply the regular IMU msg type and message topic
        self.label_msgs = {}
        self.msg_buffer = {}
        for sensor_key, sensor in self.config['sensors'].items():
            if sensor['modality'] == 'rgb':
                label_msg_type = ImageWithRGBLabels
            elif sensor['modality'] == 'lidar3d':
                label_msg_type = PointCloudWithLidar3DLabels
            elif sensor['modality'] == 'depth':
                label_msg_type = DepthImageWithDepthLabels
            elif sensor['modality'] == "imu":
                label_msg_type = sensor_msgs.msg.Imu

            if sensor['modality'] == "imu":
                labels_topic = sensor["topic_name"]
            else:
                labels_topic = sensor_key + '/labels'

            # Check if the sensor has the 'continuous' key. If it does not, default it to False
            if 'continuous' not in sensor.keys():
                sensor['continuous'] = False

            # Create an empty list of data for each sensor with continuous collection for the msg_buffer
            if sensor['continuous'] == True:
                self.msg_buffer[sensor_key] = []
                print('Created message buffer for continuous data collection for sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL + "!")


            print("Waiting for first message on topic " +
                  labels_topic + ' ... ', end='')
            self.label_msgs[sensor_key] = rospy.wait_for_message(
                labels_topic, label_msg_type)
            print('received!')

        # Create an additional_data dictionary
        # Also, create the additional_data_msgs dictionary and fill it with the first messages
        if 'additional_data' in self.config:
            for description, value in self.config['additional_data'].items():
                data_dict = {'_name': description, 'modality': value['modality'], 'parent': value['link'],
                             'calibration_parent': value['parent_link'], 'calibration_child': value['child_link']}

                print("Waiting for message " + value['topic_name'] + ' ...')
                msg = rospy.wait_for_message(value['topic_name'], rospy.AnyMsg)
                print('... received!')

                self.additional_data_msgs[description] = msg

                connection_header = msg._connection_header['type'].split('/')
                msg_type = connection_header[1]
                print('Topic ' + value['topic_name'] + ' has type ' + msg_type)
                data_dict['topic'] = value['topic_name']
                data_dict['msg_type'] = msg_type

                self.additional_data[description] = data_dict

        # Create the labeler subscribers
        self.labeler_subscribers = {}
        for sensor_key, sensor in self.config['sensors'].items():
            if sensor['modality'] == 'rgb':
                label_msg_type = ImageWithRGBLabels
            elif sensor['modality'] == 'lidar3d':
                label_msg_type = PointCloudWithLidar3DLabels
            elif sensor['modality'] == 'depth':
                label_msg_type = DepthImageWithDepthLabels
            elif sensor['modality'] == 'imu':
                label_msg_type = sensor_msgs.msg.Imu

            if sensor["modality"] == "imu":
                labels_topic = sensor["topic_name"]
            else:
                labels_topic = sensor_key + '/labels'

            print('Setting up subscriber for ' + Fore.BLUE + sensor_key + Style.RESET_ALL +
                  ' label msgs on topic ' + Fore.GREEN + labels_topic + Style.RESET_ALL)

            self.labeler_subscribers[sensor_key] = rospy.Subscriber(
                labels_topic, label_msg_type,
                partial(self.callbackReceivedLabelMsg, sensor_key=sensor_key), queue_size=1)

        # Create the additional_data subscribers
        self.additional_data_subscribers = {}
        if 'additional_data' in self.config:
            for description, value in self.config['additional_data'].items():
                self.additional_data_subscribers[description] = rospy.Subscriber(
                    value['topic_name'], rospy.AnyMsg,
                    partial(self.callbackReceivedAdditionalDataMsg, additional_data_key=description), queue_size=1)

    def callbackReceivedAdditionalDataMsg(self, msg, additional_data_key):
        self.additional_data_msgs[additional_data_key] = msg

    def callbackReceivedLabelMsg(self, msg, sensor_key):
        # print('Received labels message for sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL)
        if self.config['sensors'][sensor_key]['continuous'] == True:
            self.msg_buffer[sensor_key].append(msg)
        
        self.label_msgs[sensor_key] = msg


    def callbackReceivedJointStateMsg(self, msg):
        # Add the joint positions to the dictionary
        for name, position in zip(msg.name, msg.position):
            self.joint_state_position_dict[name] = position

    def callbackDeleteCollection(self, request):
        print('callbackDeleteCollection service called')

        try:  # make sure we can convert the collection_name to int
            collection_name_int = int(request.collection_name)
        except ValueError as verr:  # do job to handle: s does not contain anything convertible to int
            response = atom_msgs.srv.DeleteCollectionResponse()
            response.success = False
            response.message = 'Failure. Cannot convert collection ' + \
                request.collection_name + ' to string.'
            return response
        except Exception as ex:  # do job to handle: Exception occurred while converting to int
            response = atom_msgs.srv.DeleteCollectionResponse()
            response.success = False
            response.message = 'Failure. Cannot convert collection ' + \
                request.collection_name + ' to string.'

        response = atom_msgs.srv.DeleteCollectionResponse()
        if collection_name_int in self.collections.keys():
            del self.collections[collection_name_int]
            response.success = True
            response.message = 'Collection ' + request.collection_name + ' deleted'

            # Save new dataset to json file
            D = {'sensors': self.sensors, 'additional_sensor_data': self.additional_data,
                 'collections': self.collections, 'calibration_config': self.config}
            output_file = self.output_folder + '/dataset.json'
            atom_core.dataset_io.saveAtomDataset(output_file, D)

            print(Fore.YELLOW + 'Deleted collection ' +
                  request.collection_name + Style.RESET_ALL)
        else:
            print('Failure. Collection ' +
                  request.collection_name + ' does not exist.')
            response.success = False
            response.message = 'Failure. Collection ' + \
                request.collection_name + ' does not exist.'

        return response

    def callbackSaveCollection(self, request):
        print('callbackSaveCollection service called')

        # Call internal save collection
        self.saveCollection()

        response = atom_msgs.srv.SaveCollectionResponse()
        response.success = True
        response.message = "Collection saved"
        return response

    def callbackGetDataset(self, request):
        print('callbackGetDataset service called')

        dataset_file = self.output_folder + '/dataset.json'
        try:
            file = open(dataset_file, 'r')
            dataset_stream = file.read()
            success = True
        except:
            dataset_stream = ''
            success = False

        response = atom_msgs.srv.GetDatasetResponse()
        response.dataset_json = dataset_stream
        response.success = success
        return response

    def getTransforms(self, abstract_transforms, buffer, time=None):
        # Initialize an empty dictionary that will store all the transforms for this data-stamp
        transforms_dict = {}

        if time is None:
            time = rospy.Time.now()

        for ab in abstract_transforms:  # Update all transformations

            transf = buffer.lookup_transform(ab['parent'], ab['child'], time)
            trans = [transf.transform.translation.x,
                     transf.transform.translation.y, transf.transform.translation.z]
            quat = [transf.transform.rotation.x,
                    transf.transform.rotation.y,
                    transf.transform.rotation.z,
                    transf.transform.rotation.w]

            key = generateKey(ab['parent'], ab['child'])
            transforms_dict[key] = {
                'trans': trans, 'quat': quat, 'parent': ab['parent'], 'child': ab['child']}

        return transforms_dict

    def getLabelersTimeStatistics(self):
        stamps = []  # a list of the several time stamps of the stored messages

        for sensor_key, sensor in self.sensors.items():
            stamps.append(self.label_msgs[sensor_key].header.stamp)

        max_delta = getMaxTimeDelta(stamps)
        # TODO : this is because of Andre's bag file problem. We should go back to the getAverageTime
        # For looking up transforms use average time of all sensor msgs
        average_time = getAverageTime(stamps)
        # average_time = getMaxTime(stamps)  # For looking up transforms use average time of all sensor msgs

        print('Times:')
        for stamp, sensor_key in zip(stamps, self.sensors):
            printRosTime(stamp, prefix=(sensor_key + ': '))

        return stamps, average_time, max_delta

    def saveCollection(self):

        # --------------------------------------
        # collect sensor data and labels (images, laser scans, etc)
        # --------------------------------------

        # --------------------------------------
        # Analyze message time stamps and decide if collection can be stored
        # --------------------------------------
        stamps, average_time, max_delta = self.getLabelersTimeStatistics()

        # if max_delta is None (only one sensor), continue
        if max_delta is not None:
            # times are close enough?
            if max_delta.to_sec() > float(self.config['max_duration_between_msgs']):
                rospy.logwarn('Max duration between msgs in collection is ' + str(max_delta.to_sec()) +
                              '. Not saving collection.')
                return None
            else:  # test passed
                rospy.loginfo(
                    'Max duration between msgs in collection is ' + str(max_delta.to_sec()))

        # collect all the transforms
        print('average_time=' + str(average_time))
        transforms = self.getTransforms(self.abstract_transforms,
                                        self.tf_buffer,
                                        average_time)  # use average time of label msgs

        printRosTime(average_time, "Collected transforms for time ")

        # --------------------------------------
        # Create joint dict
        # --------------------------------------
        joints_dict = {}
        if self.config['joints'] is not None:
            for config_joint_key, config_joint in self.config['joints'].items():

                # TODO should we set the position bias
                config_joint_dict = {'transform_key': None, 'position': None}

                # find joint in xacro
                found_in_urdf = False
                for urdf_joint in self.urdf_description.joints:
                    if config_joint_key == urdf_joint.name:
                        x, y, z = urdf_joint.origin.xyz
                        roll, pitch, yaw = urdf_joint.origin.rpy
                        config_joint_dict['origin_x'] = x
                        config_joint_dict['origin_y'] = y
                        config_joint_dict['origin_z'] = z
                        config_joint_dict['origin_roll'] = roll
                        config_joint_dict['origin_pitch'] = pitch
                        config_joint_dict['origin_yaw'] = yaw

                        ax, ay, az = urdf_joint.axis
                        config_joint_dict['axis_x'] = ax
                        config_joint_dict['axis_y'] = ay
                        config_joint_dict['axis_z'] = az
                        config_joint_dict['parent_link'] = urdf_joint.parent
                        config_joint_dict['child_link'] = urdf_joint.child
                        config_joint_dict['joint_type'] = urdf_joint.type
                        found_in_urdf = True
                        break

                if not found_in_urdf:
                    atomError('Defined joint ' + Fore.BLUE + config_joint_key + Style.RESET_ALL +
                              ' to be calibrated, but it does not exist in the urdf description. Run the calibration package configuration for more information.')

                # find joint in transforms pool
                for transform_key, transform in transforms.items():
                    if config_joint_dict['parent_link'] == transform['parent'] and config_joint_dict['child_link'] == transform['child']:
                        config_joint_dict['transform_key'] = transform_key
                        break

                if config_joint_dict['transform_key'] is None:
                    atomError('Defined joint ' + Fore.BLUE + config_joint_key + Style.RESET_ALL +
                              ' to be calibrated, but it does not exist in the transformation pool. Run the calibration package configuration for more information.')

                if config_joint_key not in self.joint_state_position_dict:
                    atomError('Could not get position of joint ' + Fore.BLUE + config_joint_key + Style.RESET_ALL +
                              ' from /joint_state messages.')

                # Get current joint position from the joint state message
                config_joint_dict['position'] = self.joint_state_position_dict[config_joint_key]

                joints_dict[config_joint_key] = config_joint_dict

        # --------------------------------------
        # Create all_sensor_labels_dict
        # --------------------------------------
        all_sensor_labels_dict = {}
        for pattern_key in self.config['calibration_patterns'].keys():
            all_sensor_labels_dict[pattern_key] = {}
            for sensor_key, sensor in self.sensors.items():

                if sensor['modality'] == 'imu':
                    continue

                print('Collecting data from sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL +
                      ' for pattern ' + Fore.GREEN + pattern_key + Style.RESET_ALL)

                # Search for correct pattern
                label_msg = self.label_msgs[sensor_key]
                pattern_label = None
                for label in label_msg.patterns:
                    if label.pattern_name == pattern_key:
                        pattern_label = label

                # Convert pattern_label in ROS message format to atom dataset labels dict
                if pattern_label is None:
                    print(Fore.YELLOW + 'Could not find pattern ' + pattern_key +
                          ' in label_msg of sensor ' + sensor_key + Style.RESET_ALL)
                    labels_dict = {'detected': False,
                                   'idxs': [], 'idxs_limit_points': []}
                    all_sensor_labels_dict[pattern_key][sensor_key] = labels_dict

                elif sensor['modality'] in ['rgb']:
                    labels_dict = {
                        'detected': pattern_label.detected, 'idxs': []}

                    for detection_2d in pattern_label.idxs:
                        labels_dict['idxs'].append({'id': detection_2d.id,
                                                    'x': detection_2d.x,
                                                    'y': detection_2d.y})

                    all_sensor_labels_dict[pattern_key][sensor_key] = labels_dict

                elif sensor['modality'] in ['lidar3d', 'depth']:

                    labels_dict = {'detected': pattern_label.detected,
                                   'idxs': [], 'idxs_limit_points': []}

                    for idx in pattern_label.idxs:
                        labels_dict['idxs'].append(idx)

                    for idx_limit_point in pattern_label.idxs_limit_points:
                        labels_dict['idxs_limit_points'].append(idx_limit_point)

                    all_sensor_labels_dict[pattern_key][sensor_key] = labels_dict

                else:
                    raise ValueError('Unknown message type.')

        # --------------------------------------
        # Create all_sensor_data_dict
        # --------------------------------------
        all_sensor_data_dict = {}
        for sensor_key, sensor in self.sensors.items():

            if sensor['modality'] in ['rgb']:
                msg = self.label_msgs[sensor_key].image
            elif sensor['modality'] in ['lidar3d']:
                msg = self.label_msgs[sensor_key].point_cloud
            elif sensor['modality'] in ['depth']:
                msg = self.label_msgs[sensor_key].image

            all_sensor_data_dict[sensor['_name']
                                 ] = message_converter.convert_ros_message_to_dictionary(msg)

        # --------------------------------------
        # Create all_additional_data_dict
        # --------------------------------------
        all_additional_data_dict = {}
        for additional_data_key, _ in self.additional_data.items():
            msg = self.additional_data_msgs[additional_data_key]
            all_additional_data_dict[additional_data_key] = message_converter.convert_ros_message_to_dictionary(
                msg)

        # --------------------------------------
        # Create continuous_sensor_data_dict
        # --------------------------------------
        continuous_sensor_data_dict = {}
        for sensor_key, msg_list in self.msg_buffer.items():
            continuous_sensor_data_dict[sensor_key] = []
            for msg in msg_list:
                msg_dict = message_converter.convert_ros_message_to_dictionary(msg)
                continuous_sensor_data_dict[sensor_key].append(msg_dict)
                
            
        # --------------------------------------
        # Create collection_dict
        # --------------------------------------
        collection_dict = {'data': all_sensor_data_dict, 'labels': all_sensor_labels_dict,
                           'transforms': transforms,
                           'additional_data': all_additional_data_dict, 'joints': joints_dict}

        collection_key = str(self.data_stamp).zfill(
            3)  # collection names are 000, 001, etc
        self.collections[collection_key] = collection_dict

        # Update pattern dict with transform_initial of this collection
        dataset = {'_metadata': self.metadata,
                   'calibration_config': self.config,
                   'collections': self.collections,
                   'additional_sensor_data': self.additional_data,
                   'continuous_sensor_data': continuous_sensor_data_dict,
                   'sensors': self.sensors,
                   'transforms': self.transforms,
                   'patterns': self.patterns_dict}

        print('Estimating pattern poses for collection ...')
        estimatePatternPosesForCollection(dataset, collection_key)

        self.data_stamp += 1

        # Save to json file.
        output_file = self.output_folder + '/dataset.json'
        atom_core.dataset_io.saveAtomDataset(output_file, dataset)

    def getAllAbstractTransforms(self):

        # Get a list of all transforms to collect
        transforms_list = []

        rospy.sleep(0.1)
        now = rospy.Time.now()

        # Error came up in
        # https://github.com/miguelriemoliveira/agri-gaia-test/issues/1
        # Identified that this was deprecated in
        # https://answers.ros.org/question/335327/yamlload-is-deprecated-when-source-melodic_wssetupbash-is-executed/
        # Using solution from
        # https://answers.ros.org/question/347857/how-to-get-list-of-all-tf-frames-programatically/
        # all_frames = self.listener.getFrameStrings()
        frames_dict = yaml.safe_load(
            self.listener._buffer.all_frames_as_yaml())
        all_frames = list(frames_dict.keys())

        for frame in all_frames:
            print('Waiting for transformation from ' + Fore.BLUE + frame + Style.RESET_ALL +
                  ' to ' + Fore.BLUE + self.world_link + Style.RESET_ALL + ' (max 3 secs)')
            try:
                self.listener.waitForTransform(
                    frame, self.world_link, now, rospy.Duration(3))
                chain = self.listener.chain(
                    frame, now, self.world_link, now, self.world_link)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logerr('Could not get transform from ' +
                             frame + ' to ' + self.world_link + '(max 3 secs)')
                continue

            for idx in range(0, len(chain) - 1):
                parent = chain[idx]
                child = chain[idx + 1]
                transforms_list.append(
                    {'parent': parent, 'child': child, 'key': generateKey(parent, child)})

        # https://stackoverflow.com/questions/31792680/how-to-make-values-in-list-of-dictionary-unique
        uniq_l = list(map(dict, frozenset(frozenset(i.items())
                      for i in transforms_list)))
        return uniq_l  # get unique values
