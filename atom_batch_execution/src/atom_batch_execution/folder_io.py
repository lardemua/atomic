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

def stripAutomaticSuffixes(folders,args):
    fold_suffix_size = len(args['fold_suffix']) + 3  # because the suffix is complemented by the run number as in 001
    run_suffix_size = len(args['run_suffix']) + 3  # because the suffix is complemented by the run number as in 001
    suffix_size = 0
    
    if isinstance(folders,list):

        if args['fold_suffix'] in folders[0]:
            suffix_size += fold_suffix_size
        if args['run_suffix'] in folders[0]:
            suffix_size += run_suffix_size
        experiments = list(set([x[:-suffix_size] for x in folders]))  # Remove the "_foldXX" suffix

    elif isinstance(folders,str):

        if args['fold_suffix'] in folders:
            suffix_size += fold_suffix_size
        if args['run_suffix'] in folders:
            suffix_size += run_suffix_size
        experiments = folders[:-suffix_size]  # Remove the "_foldXX" suffix

    return experiments
