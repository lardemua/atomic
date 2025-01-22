
# Standard imports
import os

import re
import subprocess

# Ros imports
from urllib.parse import urlparse

import yaml
import rospkg
from rospkg.common import ResourceNotFound
from colorama import Fore, Style

from atom_core.utilities import atomError, atomWarn
from atom_core.system import resolvePath, expandToLaunchEnv


def dictionaries_have_same_keys(d1, d2):

    if not d1.keys() == d2.keys():  # Check if all high level keys exist

        extra_keys = []
        for d1_key in d1.keys():
            if not d1_key in d2.keys():
                extra_keys.append(d1_key)

        missing_keys = []
        for d2_key in d2.keys():
            if not d2_key in d1.keys():
                missing_keys.append(d2_key)

        return False, extra_keys, missing_keys

    else:
        return True, [], []


def verifyConfig(config, template_config):

    # Check if all high level keys exist
    same_keys, extra_keys, missing_keys = dictionaries_have_same_keys(config, template_config)
    if not same_keys:
        atomError('Config file does not have the correct keys.\nKeys that should not exist: ' +
                  str(extra_keys) + '\nKeys that are missing : ' + str(missing_keys))

    sensor_template = template_config['sensors']['hand_camera']
    for sensor_key, sensor in config['sensors'].items():
        same_keys, extra_keys, missing_keys = dictionaries_have_same_keys(sensor, sensor_template)
        if not same_keys:
            if missing_keys == ['continuous']:
                atomWarn('In config file, sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL + ' does not have the "continuous" key. Disbaling continuous data collection.')
            else:
                atomError('In config file, sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL + ' does not have the correct keys.\nKeys that should not exist: ' +
                      str(extra_keys) + '\nKeys that are missing : ' + str(missing_keys))

    if config['additional_tfs'] is not None:
        additional_tf_template = {'parent_link': None, 'child_link': None}
        for additional_tf_key, additional_tf in config['additional_tfs'].items():
            same_keys, extra_keys, missing_keys = dictionaries_have_same_keys(additional_tf, additional_tf_template)
            if not same_keys:
                if missing_keys == ['continuous']:
                    atomWarn('In config file, sensor ' + Fore.BLUE + sensor_key + Style.RESET_ALL + ' does not have the "continuous" key. Disabling continuous data collection.')
                else:
                    atomError('In config file, additional_tf ' + Fore.BLUE + additional_tf_key + Style.RESET_ALL + ' does not have the correct keys.\nKeys that should not exist: ' +
                          str(extra_keys) + '\nKeys that are missing : ' + str(missing_keys))

    if config['joints'] is not None:
        joint_template = {'params_to_calibrate': None}
        for joint_key, joint in config['joints'].items():
            same_keys, extra_keys, missing_keys = dictionaries_have_same_keys(joint, joint_template)
            if not same_keys:
                atomError('In config file, joint ' + Fore.BLUE + joint_key + Style.RESET_ALL + ' does not have the correct keys.\nKeys that should not exist: ' +
                          str(extra_keys) + '\nKeys that are missing : ' + str(missing_keys))

    calibration_pattern_template = template_config['calibration_patterns']['pattern_1']
    for calibration_pattern_key, calibration_pattern in config['calibration_patterns'].items():
        same_keys, extra_keys, missing_keys = dictionaries_have_same_keys(
            calibration_pattern, calibration_pattern_template)
        if not same_keys:
            atomError('In config file, calibration_pattern ' + Fore.BLUE + calibration_pattern_key + Style.RESET_ALL + ' does not have the correct keys.\nKeys that should not exist: ' +
                      str(extra_keys) + '\nKeys that are missing : ' + str(missing_keys))


def loadConfig(filename, check_paths=True):
    config = loadYMLConfig(filename)
    if config is None:
        raise ValueError(Fore.RED + 'Your config file ' + filename +
                         ' could not be read. Aborting.' + Fore.RESET)

    # if "robot_name" not in config.keys():  # in config:
    #     raise ValueError(Fore.RED +
    #         'Error: argument robot_name is missing in config.yaml'+ Style.RESET_ALL)
    # Check if config has all the necessary keys.
    rospack = rospkg.RosPack()
    template_file = rospack.get_path('atom_calibration') + '/templates/config.yml'
    template_config = loadYMLConfig(template_file)
    missing_parameters = verifyConfig(config, template_config)
    # print(missing_parameters)

    if missing_parameters:  # list is not empty
        print('Your config file ' + Fore.BLUE + filename + Style.RESET_ALL +
              ' appears to be corrupted. These mandatory parameters are missing: ' + Fore.BLUE +
              str(missing_parameters) + Style.RESET_ALL + '\nPerhaps your file format is not updated.\n' +
              'To fix, check scripts ' + Fore.BLUE + 'add_joints_to_config_file' + Style.RESET_ALL + ' and ' +
              Fore.BLUE + 'add_package_name_to_config_file' + Style.RESET_ALL + ' in atom_calibration/scripts/utilities')
        exit(0)

    # Check if description file is ok
    # print(config['description_file'])
    fullpath, name, uri = uriReader(config['description_file'])

    # Check if bag_file is ok
    fullpath, name, uri = uriReader(config['bag_file'])

    # Check if calibration_pattern/mesh_file is ok

    for calibration_pattern_key, calibration_pattern in config['calibration_patterns'].items():
        fullpath, name, uri = uriReader(calibration_pattern['mesh_file'])

    return config


def loadYMLConfig(filename):
    """Load configuration from a yml file"""
    try:
        with open(filename, 'r') as f:
            obj = yaml.load(f, Loader=yaml.SafeLoader)
    except OSError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
        return None

    return obj


def validateLinks(world_link, sensors, urdf):
    try:
        for name, sensor in sensors.items():
            chain = urdf.get_chain(world_link, sensor.link)
            if sensor.parent_link not in chain or sensor.child_link not in chain:
                print("{}: The links '{}' and '{}' are not part of the same chain.".format(sensor.name,
                                                                                           sensor.parent_link,
                                                                                           sensor.child_link))
                return False
    except KeyError as e:
        link_name = str(e).strip("'")
        if link_name == urdf.get_root():
            print("Configuration contains an unknown base link: {}".format(world_link))
            return False

        print("Configuration contains an unknown link: {}".format(link_name))
        return False

    return True


def uriReader(resource):
    uri = urlparse(str(resource))
    # print(uri)
    if uri.scheme == 'package':  # using a ros package uri
        # print('This is a ros package')
        rospack = rospkg.RosPack()
        try:
            rospack.get_path(uri.netloc)
        except ResourceNotFound:
            atomError('Package ' + uri.netloc + ' does not exist.')

        fullpath = resolvePath(rospack.get_path(uri.netloc) + uri.path)
        relpath = '$(find {}){}'.format(uri.netloc, uri.path)

    elif uri.scheme == 'file':  # local file
        # print('This is a local file')
        fullpath = resolvePath(uri.netloc + uri.path)
        relpath = fullpath
    elif uri.scheme == '':  # no scheme, assume local file
        # print('This is a local file')

        fullpath = resolvePath(uri.path)
        relpath = expandToLaunchEnv(uri.path)
    else:
        raise ValueError('Cannot parse resource "' + resource + '", unknown scheme "' + uri.scheme + '".')

    if not os.path.exists(fullpath):
        atomError(Fore.BLUE + fullpath + Style.RESET_ALL + ' does not exist.')

    return fullpath, os.path.basename(fullpath), relpath
