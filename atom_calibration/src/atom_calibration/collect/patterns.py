from copy import deepcopy
import math
from colorama import Fore, Style
import cv2
import numpy as np
from atom_core.drawing import drawTextOnImage
from atom_core.utilities import atomError, atomWarn

from tf import transformations
from atom_core.geometry import traslationRodriguesToTransform
from atom_calibration.collect import patterns as opencv_patterns

import atom_core.atom


class ChessboardPattern(object):
    def __init__(self, size, length):
        self.size = (size["x"], size["y"])
        self.length = length

    def detect(self, image, equalize_histogram=False):

        if len(image.shape) == 3:  # convert to gray if it is an rgb image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if equalize_histogram:
            gray = cv2.equalizeHist(gray)

        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(gray, self.size)
        if not found:
            return {"detected": False, 'keypoints': corners, 'ids': []}

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.0001)
        sub_pixel_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        sub_pixel_corners = corners

        return {"detected": True, 'keypoints': sub_pixel_corners, 'ids': range(0, len(sub_pixel_corners))}

    def drawKeypoints(self, image, result, length=0.2, color=(1, 0, 0), K=None, D=None, pattern_name=None, debug=False):

        if result['keypoints'] is None or len(result['keypoints']) == 0:
            return

        if not result['detected']:
            return

        for point in result['keypoints']:
            cv2.drawMarker(image, (int(point[0][0]), int(point[0][1])), color, cv2.MARKER_CROSS, 14)
            cv2.circle(image, (int(point[0][0]), int(point[0][1])), 7, color, lineType=cv2.LINE_AA)

        if K is not None and D is not None:  # estimate pose and draw axis on image

            # Must convert from dictionary back to opencv strange np format just to show
            objp = np.zeros((self.size[0] * self.size[1], 3), np.float32)
            # TODO only works for first pattern
            objp[:, :2] = self.length * np.mgrid[0:self.size[0], 0:self.size[1]].T.reshape(-1, 2)

            # Build a numpy array with the chessboard corners
            corners = np.zeros((len(result['keypoints']), 1, 2), dtype=float)
            ids = list(range(0, len(result['keypoints'])))

            points = result['keypoints'].astype(np.int32)
            for idx, (point, id) in enumerate(zip(result['keypoints'], result['ids'])):
                corners[idx, 0, 0] = point[0][0]
                corners[idx, 0, 1] = point[0][1]
                ids[idx] = id

            np_cids = np.array(ids, dtype=int).reshape((len(result['keypoints']), 1))
            np_ccorners = np.array(corners, dtype=np.float32)

            _, rvecs, tvecs = cv2.solvePnP(
                objp[ids], np.array(corners, dtype=np.float32), K, D)

            cv2.drawFrameAxes(image, K, D, rvecs, tvecs, length)

        if pattern_name is not None:
            point = (int(corners[0][0][0]), int(corners[0][0][1]))
            drawTextOnImage(image, pattern_name,
                            font=cv2.FONT_HERSHEY_PLAIN,
                            font_scale=4.5,
                            font_thickness=3,
                            position=point,
                            text_color=color,
                            text_color_bg=(255, 255, 255))


class CharucoPattern(object):
    def __init__(self, size, length, marker_length, dictionary='DICT_5X5_100'):

        # string to charuco dictionary conversion
        charuco_dict = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000
        }

        if dictionary in charuco_dict:
            charuco_dictionary = charuco_dict[dictionary]
        else:
            atomError('Invalid dictionary set on json configuration file. Using the default DICT_5X5_100.')

        self.size = (size["x"], size["y"])
        self.number_of_corners = size["x"] * size["y"]

        if cv2.__version__ == '4.6.0':
            self.dictionary = cv2.aruco.Dictionary_get(charuco_dictionary)
            self.board = cv2.aruco.CharucoBoard_create(size["x"] + 1, size["y"] + 1, length, marker_length,
                                                       self.dictionary)
            # self.image_board = self.board.draw((787, 472))
        else:  # all versions from 4.7.0 onward
            self.dictionary = cv2.aruco.getPredefinedDictionary(charuco_dictionary)
            self.board = cv2.aruco.CharucoBoard((size["x"] + 1, size["y"] + 1), length, marker_length,
                                                self.dictionary)

    def detect(self, image, equalize_histogram=False):

        if len(image.shape) == 3:  # convert to gray if it is an rgb image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if equalize_histogram:  # equalize image histogram
            gray = cv2.equalizeHist(gray)

        # https://github.com/lardemua/atom/issues/629
        if cv2.__version__ == '4.6.0':
            params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=params)

            # Debug
            # image_gui = deepcopy(image)
#             cv2.aruco.drawDetectedMarkers(image_gui, corners, ids)
#             cv2.namedWindow('Debug image', cv2.WINDOW_NORMAL)
#             cv2.imshow('Debug image', image_gui)
#
#             cv2.namedWindow('image board', cv2.WINDOW_NORMAL)
#             cv2.imshow('image board', self.image_board)
#
#             cv2.waitKey(30)

            # cv2.aruco.refineDetectedMarkers(gray, self.board, corners, ids, rejected)
        else:
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(self.dictionary, params)
            corners, ids, rejected = detector.detectMarkers(gray)

        if len(corners) <= 8:  # Must have more than 3 corner detections
            return {"detected": False, 'keypoints': np.array([]), 'ids': []}

        # Interpolation of charuco corners
        ret, ccorners, cids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)

        # Produce results dictionary -------------------------

        if ccorners is None:  # Must have interpolation running ok
            return {"detected": False, 'keypoints': np.array([]), 'ids': []}

        # A valid detection must have at least 25% of the total number of corners.
        if len(ccorners) <= self.number_of_corners / 4:
            return {"detected": False, 'keypoints': np.array([]), 'ids': []}

        # If all above works, return detected corners.
        return {'detected': True, 'keypoints': ccorners, 'ids': cids.ravel().tolist()}

    def drawKeypoints(self, image, result, length=0.2, color=(255, 0, 0), K=None, D=None, pattern_name=None, debug=False):
        if result['keypoints'] is None or len(result['keypoints']) == 0:
            return

        if not result['detected']:
            return

        # Must convert from dictionary back to opencv strange np format just to show
        # Build a numpy array with the chessboard corners
        ccorners = np.zeros((len(result['keypoints']), 1, 2), dtype=float)
        cids = list(range(0, len(result['keypoints'])))
        num_corners = len(cids)

        points = result['keypoints'].astype(np.int32)
        for idx, (point, id) in enumerate(zip(result['keypoints'], result['ids'])):
            ccorners[idx, 0, 0] = point[0][0]
            ccorners[idx, 0, 1] = point[0][1]
            cids[idx] = id

        np_cids = np.array(cids, dtype=int).reshape((len(result['keypoints']), 1))
        np_ccorners = np.array(ccorners, dtype=np.float32)

        # Draw charuco corner detection
        if debug:
            image = cv2.aruco.drawDetectedCornersCharuco(image, np_ccorners, np_cids, color)

        if K is not None and D is not None:  # estimate pose and draw axis on image
            rvecs, tvecs = None, None
            _, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(np_ccorners,
                                                                 np_cids, self.board,
                                                                 K, D, rvecs, tvecs)
            # Draw frame on the image
            cv2.drawFrameAxes(image, K, D, rvecs, tvecs, length)

        if pattern_name is not None:
            point = tuple(ccorners[0][0])

            drawTextOnImage(image, pattern_name,
                            font=cv2.FONT_HERSHEY_PLAIN,
                            font_scale=4.5,
                            font_thickness=3,
                            position=point,
                            text_color=color,
                            text_color_bg=(255, 255, 255))


def initializePatternsDict(config, step=0.02):
    """
    Creates the necessary data related to the calibration pattern
    :return: a patterns dictionary
    """

    patterns_dict = {}
    for pattern_key, pattern in config['calibration_patterns'].items():

        nx = pattern['dimension']['x']
        ny = pattern['dimension']['y']
        square = pattern['size']
        # Border can be a scalar or {'x': ..., 'y': ...}
        if type(pattern['border_size']) is dict:
            border_x = pattern['border_size']['x']
            border_y = pattern['border_size']['y']
        else:
            border_x = border_y = pattern['border_size']

        pattern_dict = {  # All coordinates in the pattern's local coordinate system. Since z=0 for all points, it is omitted.
            # [{'idx': 0, 'x': 3, 'y': 4}, ..., ] # Pattern's visual markers
            'corners': [],
            'frame': {'corners': {'top_left': {'x': 0, 'y': 0},  # Physical outer boundaries of the pattern
                                  'top_right': {'x': 0, 'y': 0},
                                  'bottom_right': {'x': 0, 'y': 0},
                                  'bottom_left': {'x': 0, 'y': 0}},
                      'lines_sampled': {'top': [],  # [{'x':0, 'y':0}]
                                        'bottom': [],  # [{'x':0, 'y':0}]
                                        'left': [],  # [{'x':0, 'y':0}]
                                        'right': []}  # [{'x':0, 'y':0}]
                      },
            # Transitions from black to white squares. Just a set of sampled points
            'transitions': {'vertical': [],  # [{'x': 3, 'y': 4}, ..., {'x': 30, 'y': 40}]},
                            'horizontal': []},  # [{'x': 3, 'y': 4}, ..., {'x': 30, 'y': 40}]},
            'transforms_initial': {},  # {'collection_key': {'trans': ..., 'quat': 4}, ...}
        }

        if pattern['pattern_type'] == 'chessboard':
            # Chessboard: Origin on top left corner, X left to right, Y top to bottom

            # ---------------- Corners ----------------
            # idx left to right, top to bottom
            idx = 0
            for row in range(0, pattern['dimension']['y']):
                for col in range(0, pattern['dimension']['x']):
                    pattern_dict['corners'].append({'id': idx, 'x': col * square, 'y': row * square})
                    idx += 1

            # ---------------- Frame ----------------
            # Corners
            pattern_dict['frame']['corners']['top_left'] = {
                'x': -square - border_x, 'y': -square - border_y}
            pattern_dict['frame']['corners']['top_right'] = {
                'x': nx * square + border_x, 'y': -square - border_y}
            pattern_dict['frame']['corners']['bottom_right'] = {
                'x': nx * square + border_x, 'y': ny * square + border_y}
            pattern_dict['frame']['corners']['bottom_left'] = {
                'x': -square - border_x, 'y': ny * square + border_y}

            # Lines sampled
            pattern_dict['frame']['lines_sampled']['top'] = sampleLineSegment(
                pattern_dict['frame']['corners']['top_left'], pattern_dict['frame']['corners']['top_right'], step)
            pattern_dict['frame']['lines_sampled']['bottom'] = sampleLineSegment(
                pattern_dict['frame']['corners']['bottom_left'], pattern_dict['frame']['corners']['bottom_right'], step)
            pattern_dict['frame']['lines_sampled']['left'] = sampleLineSegment(pattern_dict['frame']['corners']['top_left'],
                                                                               pattern_dict['frame']['corners']['bottom_left'],
                                                                               step)
            pattern_dict['frame']['lines_sampled']['right'] = sampleLineSegment(
                pattern_dict['frame']['corners']['top_right'], pattern_dict['frame']['corners']['bottom_right'], step)

            # -------------- Transitions ----------------
            # vertical
            for col in range(0, config['calibration_patterns'][pattern_key]['dimension']['x']):
                p0 = {'x': col * square, 'y': 0}
                p1 = {'x': col * square, 'y': (ny - 1) * square}
                pts = sampleLineSegment(p0, p1, step)
                pattern_dict['transitions']['vertical'].extend(pts)

            # horizontal
            for row in range(0, config['calibration_patterns'][pattern_key]['dimension']['y']):
                p0 = {'x': 0, 'y': row * square}
                p1 = {'x': (nx - 1) * square, 'y': row * square}
                pts = sampleLineSegment(p0, p1, step)
                pattern_dict['transitions']['horizontal'].extend(pts)

            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(patterns)
            # exit(0)

        elif pattern['pattern_type'] == 'charuco':
            # Charuco: Origin on bottom left corner, X left to right, Y bottom to top

            # ---------------- Corners ----------------
            # idx left to right, bottom to top
            idx = 0
            for row in range(0, pattern['dimension']['y']):
                for col in range(0, pattern['dimension']['x']):
                    pattern_dict['corners'].append(
                        {'id': idx, 'x': square + col * square, 'y': square + row * square})
                    idx += 1

            # ---------------- Frame ----------------
            # Corners
            pattern_dict['frame']['corners']['top_left'] = {
                'x': -border_x, 'y': - border_y}
            pattern_dict['frame']['corners']['top_right'] = {
                'x': square + nx * square + border_x, 'y': -border_y}
            pattern_dict['frame']['corners']['bottom_right'] = {
                'x': square + nx * square + border_x, 'y': ny * square + border_y + square}
            pattern_dict['frame']['corners']['bottom_left'] = {
                'x': - border_x, 'y': ny * square + border_y + square}

            # Lines sampled
            pattern_dict['frame']['lines_sampled']['top'] = sampleLineSegment(
                pattern_dict['frame']['corners']['top_left'], pattern_dict['frame']['corners']['top_right'], step)
            pattern_dict['frame']['lines_sampled']['bottom'] = sampleLineSegment(
                pattern_dict['frame']['corners']['bottom_left'], pattern_dict['frame']['corners']['bottom_right'], step)
            pattern_dict['frame']['lines_sampled']['left'] = sampleLineSegment(pattern_dict['frame']['corners']['top_left'],
                                                                               pattern_dict['frame']['corners']['bottom_left'],
                                                                               step)
            pattern_dict['frame']['lines_sampled']['right'] = sampleLineSegment(
                pattern_dict['frame']['corners']['top_right'], pattern_dict['frame']['corners']['bottom_right'], step)

            # -------------- Transitions ----------------
            # vertical
            for col in range(0, pattern['dimension']['x']):
                p0 = {'x': col * square, 'y': 0}
                p1 = {'x': col * square, 'y': (ny - 1) * square}
                pts = sampleLineSegment(p0, p1, step)
                pattern_dict['transitions']['vertical'].extend(pts)

            # horizontal
            for row in range(0, pattern['dimension']['y']):
                p0 = {'x': 0, 'y': row * square}
                p1 = {'x': (nx - 1) * square, 'y': row * square}
                pts = sampleLineSegment(p0, p1, step)
                pattern_dict['transitions']['horizontal'].extend(pts)
        else:
            raise ValueError(
                'Unknown pattern type: ' + pattern['pattern_type'])

        patterns_dict[pattern_key] = pattern_dict  # add this pattern to the patterns dict

    return patterns_dict


def estimatePatternPosesForCollection(dataset, collection_key):

    collection = dataset['collections'][collection_key]
    for pattern_key, pattern in dataset['calibration_config']['calibration_patterns'].items():

        nx = pattern['dimension']['x']
        ny = pattern['dimension']['y']
        square = pattern['size']

        # -----------------------------------
        # Create first guess for pattern pose
        # -----------------------------------
        size = {'x': pattern['dimension']['x'], 'y': pattern['dimension']['y']}
        length = pattern['size']
        inner_length = pattern['inner_size']
        dictionary = pattern['dictionary']

        if pattern['pattern_type'] == 'charuco':
            opencv_pattern = opencv_patterns.CharucoPattern(size, length, inner_length, dictionary)

        flg_detected_pattern = False

        dataset['patterns'][pattern_key]['transforms_initial'][collection_key] = {
            'detected': False}  # by default no detection

        for sensor_key, sensor in dataset['sensors'].items():

            # Ignore the sensor if it's an IMU, as no pattern pose estimation can be made
            if sensor['modality'] == 'imu':
                continue

            # if pattern not detected by sensor in collection
            if not collection['labels'][pattern_key][sensor_key]['detected']:
                continue

            # change accordingly to the first camera to give chessboard first poses
            if sensor['modality'] == 'rgb':
                K = np.ndarray((3, 3), dtype=float, buffer=np.array(
                    sensor['camera_info']['K']))
                D = np.ndarray((5, 1), dtype=float, buffer=np.array(
                    sensor['camera_info']['D']))

                # TODO should we not read these from the dictionary?
                objp = np.zeros((nx * ny, 3), np.float32)
                objp[:, :2] = square * np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

                # Build a numpy array with the charuco corners
                corners = np.zeros((len(collection['labels'][pattern_key][sensor_key]['idxs']), 1, 2),
                                   dtype=float)
                ids = list(range(0, len(collection['labels'][pattern_key][sensor_key]['idxs'])))
                for idx, point in enumerate(collection['labels'][pattern_key][sensor_key]['idxs']):
                    corners[idx, 0, 0] = point['x']
                    corners[idx, 0, 1] = point['y']
                    ids[idx] = point['id']

                # Find pose of the camera w.r.t the chessboard
                np_ids = np.array(ids, dtype=int)
                rvecs, tvecs = None, None

                # TODO only works for first pattern
                if pattern['pattern_type'] == 'charuco':
                    _, rvecs, tvecs = cv2.aruco.estimatePoseCharucoBoard(np.array(corners, dtype=np.float32),
                                                                         np_ids, opencv_pattern.board,
                                                                         K, D, rvecs, tvecs)
                else:
                    _, rvecs, tvecs = cv2.solvePnP(objp[ids], np.array(corners, dtype=np.float32), K, D)

                # Compute the pose of the pattern w.r.t the pattern parent link
                root_T_sensor = atom_core.atom.getTransform(
                    pattern['parent_link'],
                    sensor['camera_info']['header']['frame_id'], collection['transforms'])

                sensor_T_pattern = traslationRodriguesToTransform(
                    tvecs, rvecs)
                root_T_chessboard = np.dot(root_T_sensor, sensor_T_pattern)
                T = deepcopy(root_T_chessboard)
                T[0:3, 3] = 0  # remove translation component from 4x4 matrix

                # print('Creating first guess for collection ' + collection_key + ' using sensor ' + sensor_key)
                dataset['patterns'][pattern_key]['transforms_initial'][collection_key] = {
                    'detected': True, 'sensor': sensor_key,
                    'parent': pattern['parent_link'], 'child': pattern['link'],
                    'trans': list(root_T_chessboard[0: 3, 3]),
                    'quat': list(transformations.quaternion_from_matrix(T)), }

                flg_detected_pattern = True
                break  # don't search for this collection's chessboard on anymore sensors

        if not flg_detected_pattern:  # Abort when the chessboard is not detected by any camera on this collection
            atomWarn('Pattern ' + Fore.GREEN + pattern_key + Style.RESET_ALL + ' not detected in collection ' +
                     Fore.BLUE + collection_key + Style.RESET_ALL + '. Cannot produce initial estimate of pose.')


def sampleLineSegment(p0, p1, step):
    norm = math.sqrt((p1['x'] - p0['x']) ** 2 + (p1['y'] - p0['y']) ** 2)
    n = round(norm / step)
    vector_x = p1['x'] - p0['x']
    vector_y = p1['y'] - p0['y']
    pts = []
    for alfa in np.linspace(0, 1, num=n, endpoint=True, retstep=False, dtype=float):
        x = p0['x'] + vector_x * alfa
        y = p0['y'] + vector_y * alfa
        pts.append({'x': x, 'y': y})
    return pts
