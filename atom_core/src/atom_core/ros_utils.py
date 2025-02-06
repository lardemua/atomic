import itertools
import rospy

# This import must be here although it is not explicitly required by the code. It will be required in
# getMessageTypeFromTopic(topic) when the eval(msg_type_str) is executed. To avoid having automated import optimization
# mechanisms remove the import, we declare tmp variables just to explicitly use the imports. If you have any better
# ideas ...
# TODO find a better way
from sensor_msgs.msg import PointCloud2, Image, LaserScan
tmp_image = Image()
tmp_pointcloud = PointCloud2()
tmp_laserscan = LaserScan()


def filterLaunchArguments(argvs):
    # Roslaunch files send a "__name:=..." argument (and __log:=) which disrupts the argparser. The solution is to
    # filter this argv. in addition, the first argument is the node name, which should also not be given to the
    # parser.

    argvs_filtered = []
    for i, argv in enumerate(argvs):
        if (not all(x in argv for x in ['__', ':='])) and (i != 0):
            argvs_filtered.append(argv)

    return argvs_filtered


def getMessageTypeFromTopic(topic):
    # Wait for a message to infer the type
    # http://schulz-m.github.io/2016/07/18/rospy-subscribe-to-any-msg-type/
    rospy.loginfo('Waiting for first message on topic ' + topic + ' ...')
    msg = rospy.wait_for_message(topic, rospy.AnyMsg)

    connection_header = msg._connection_header['type'].split('/')
    # ros_pkg = connection_header[0] + '.msg'
    msg_type_str = connection_header[1]
    msg_type = eval(msg_type_str)

    return msg_type_str, msg_type


def printRosTime(time, prefix=""):
    print(prefix + str(time.secs) + "." + str(time.nsecs))


def getMaxTimeDelta(stamps):
    if len(stamps) < 2:  # need at least two time stamps to compute a delta
        return None

    pairs = list(itertools.combinations(stamps, 2))
    max_duration = rospy.Duration(0)
    for p1, p2 in pairs:
        d = abs(p1 - p2)
        if d > max_duration:
            max_duration = d

    return max_duration


def getMaxTime(stamps):
    reference_time = rospy.Time.now()  # get a time at the start of this call
    durations = [abs((stamp - reference_time).to_sec()) for stamp in stamps]

    max_duration = max(durations)
    max_time = reference_time + rospy.Duration(max_duration)

    printRosTime(reference_time, "reference_time: ")
    printRosTime(max_time, "max_time: ")
    print("durations = " + str(durations))
    print("max_duration = " + str(max_duration))

    return max_time


def getAverageTime(stamps):

    reference_time = rospy.Time.now()  # get a time at the start of this call

    if len(stamps) < 1:  # need at least two time stamps to compute a delta
        return reference_time

    durations = [(stamp - reference_time).to_sec() for stamp in stamps]
    avg_duration = sum(durations) / len(durations)
    avg_time = reference_time + rospy.Duration(avg_duration)

    printRosTime(reference_time, "reference_time: ")
    printRosTime(avg_time, "avg_time: ")
    print("durations = " + str(durations))
    print("avg_duration = " + str(avg_duration))

    return avg_time
