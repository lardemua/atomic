# Standard imports
import os
from datetime import datetime
from colorama import Fore, Style
from copy import deepcopy

# Ros imports
import rospkg
import atom_core
from atom_core.utilities import atomError
import tf

# Atom imports
from urdf_parser_py.urdf import Pose as URDFPose
from urdf_parser_py.urdf import URDF, Link, Joint, Mesh, Visual, JointCalibration
from atom_core.system import execute
from atom_core.config_io import uriReader
from atom_core.naming import generateKey


def readXacroFile(description_file):
    # xml_robot = URDF.from_parameter_server()
    urdf_file = '/tmp/description.urdf'
    # print('Parsing description file ' + description_file)
    execute('xacro ' + description_file + ' -o ' + urdf_file,
            verbose=False)  # create a temp urdf file
    try:
        xml_robot = URDF.from_xml_file(urdf_file)  # read teh urdf file
    except:
        raise ValueError('Could not parse description file ' + description_file)

    return xml_robot


def saveResultsXacro(dataset, selected_collection_key, transforms_list, verbose=False):
    # Cycle all sensors in calibration config, and for each replace the optimized transform in the original xacro
    # Parse xacro description file
    description_file, _, _ = uriReader(dataset["calibration_config"]["description_file"])

    xml_robot = readXacroFile(description_file)

    # Put transformations from sensors and additional_tfs into the urdf
    for transform_key in transforms_list:

        # NOTE This is only valid if the generateTransformKey was called with argument separator='-'
        parent = transform_key.split('-')[0]
        child = transform_key.split('-')[1]

        trans = list(dataset["collections"][selected_collection_key]
                     ["transforms"][transform_key]["trans"])
        quat = list(dataset["collections"][selected_collection_key]
                    ["transforms"][transform_key]["quat"])
        found = False

        for joint in xml_robot.joints:
            if joint.parent == parent and joint.child == child:
                found = True

                # if origin is None create a new URDFPose.
                # See https://github.com/lardemua/atom/issues/559
                if joint.origin is None:  #
                    joint.origin = URDFPose()

                # print("Replacing xyz = " + str(joint.origin.xyz) + " by " + str(trans))
                joint.origin.xyz = trans

                rpy = list(tf.transformations.euler_from_quaternion(quat, axes="sxyz"))
                # print("Replacing rpy = " + str(joint.origin.rpy) + " by " + str(rpy))
                joint.origin.rpy = rpy

                if verbose:
                    print(parent)
                    print('Saving transform key ' + transform_key + ' to joint ' +
                          joint.name + ' with parent ' + parent + ' and child ' + child)
                break

        if not found:
            continue  # see $897

    # Output optimized joints
    calibration_config = dataset['calibration_config']
    if calibration_config['joints'] is not None:
        for joint_key, joint_dict in dataset['collections'][selected_collection_key]['joints'].items():
            found = False

            for joint in xml_robot.joints:
                if joint_key != joint.name:
                    continue

                joint.origin.xyz = [
                    joint_dict['origin_x'],
                    joint_dict['origin_y'],
                    joint_dict['origin_z']]
                joint.origin.rpy = [
                    joint_dict['origin_roll'],
                    joint_dict['origin_pitch'],
                    joint_dict['origin_yaw']]
                found = True
                if verbose:
                    print('Saving optimized joint key ' + joint_key + ' to xacro joint ' +
                          joint.name + ' with parent ' + joint.parent + ' and child ' + joint.child)
                break

            if not found:
                atomError('Could not find optimized joint ' + joint_key + ' in urdf')

    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_name = "optimized_" + time + ".urdf.xacro"

    rospack = rospkg.RosPack()

    package_name = dataset["_metadata"]["package_name"]

    path_to_urdf_directory = (rospack.get_path(package_name) + "/urdf/")
    # path_to_optimized_directory = (path_to_urdf_directory + "optimized/")

    # if not os.path.exists(path_to_optimized_directory):
    #     os.mkdir(path_to_optimized_directory)
    # filename_results_xacro = path_to_optimized_directory + file_name
    # with open(filename_results_xacro, "w") as out:
    #     out.write(URDF.to_xml_string(xml_robot))

    # Save the urdf without the pattern
    optimized_urdf_file = path_to_urdf_directory + 'optimized.urdf.xacro'
    with open(optimized_urdf_file, "w", ) as out:
        out.write(URDF.to_xml_string(xml_robot))
        # print("Saving optimized.urdf.xacro in " + filename_results_xacro + ".")

    print("Optimized xacro saved to " + str(optimized_urdf_file) +
          " . You can use it as a ROS robot_description.")

    # Save optimized xacro with patterns
    for pattern_key, pattern in dataset['calibration_config']['calibration_patterns'].items():
        if not pattern['fixed']:
            continue
        pattern_mesh = Mesh(filename=pattern['mesh_file'])
        pattern_visual = Visual(name=pattern_key + '_visual',
                                geometry=pattern_mesh,
                                origin=URDFPose(xyz=[0, 0, 0], rpy=[0, 0, 0]))
        pattern_link = Link(name=pattern['link'],
                            visual=pattern_visual,
                            origin=URDFPose(xyz=[0, 0, 0], rpy=[0, 0, 0]))
        xml_robot.add_link(pattern_link)

        transform_key = generateKey(pattern['parent_link'], pattern['link'], suffix='')
        trans = list(dataset["collections"][selected_collection_key]
                     ["transforms"][transform_key]["trans"])
        quat = list(dataset["collections"][selected_collection_key]
                    ["transforms"][transform_key]["quat"])
        rpy = list(tf.transformations.euler_from_quaternion(quat, axes="sxyz"))

        pattern_joint = Joint(name=pattern['parent_link'] + '-' + pattern['link'],
                              parent=pattern['parent_link'],
                              child=pattern['link'],
                              joint_type='fixed',
                              origin=URDFPose(xyz=trans, rpy=rpy))

        xml_robot.add_joint(pattern_joint)

        if verbose:
            print('Saving optimized pattern key ' + pattern_key +
                  ' to xacro by creating a new joint ' + pattern_joint.name + ' with parent ' +
                  pattern_joint.parent + ' and child ' + pattern_joint.child)

    # Save the urdf with the pattern
    optimized_w_pattern_urdf_file = path_to_urdf_directory + 'optimized_w_pattern.urdf.xacro'
    with open(optimized_w_pattern_urdf_file, "w", ) as out:
        out.write(URDF.to_xml_string(xml_robot))

    # print("Optimized xacro with pattern saved to " + str(optimized_w_pattern_urdf_file) +
    #       " . You can use it as a ROS robot_description.")

    print("You can use it as a ROS robot_description by launching:\n" + Fore.BLUE + 'roslaunch ' +
          package_name + ' playbag.launch optimized:=true' + Style.RESET_ALL)
