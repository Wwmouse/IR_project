#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

depth_data = None
rgb_data = None

def read_depth(data):
    global depth_data

    # # Convert image to OpenCV format
    # try:
    #     bridge = CvBridge()
    #     depth_data = bridge.imgmsg_to_cv2(data, "passthrough")
    # except CvBridgeError as e:
    #     print(e)

    data_str = str(data)
    idx = data_str.find('[')
    img = data_str[idx+1:-1].split(', ')
    depth_data = np.array(img, dtype=np.int8)
    depth_data = depth_data.reshape(480, 640, 4)

def read_rgb(data):
    global rgb_data

    # # Convert image to OpenCV format
    # try:
    #     bridge = CvBridge()
    #     rgb_data = bridge.imgmsg_to_cv2(data, "bgr8")
    # except CvBridgeError as e:
    #     print(e)

    data_str = str(data)
    idx = data_str.find('[')
    img = data_str[idx+1:-1].split(', ')
    rgb_data = np.array(img, dtype=np.int8)
    rgb_data = rgb_data.reshape(480, 640, 3)

def read_image():
    rospy.init_node('read_image', anonymous=True)

    rospy.Subscriber('/camera/depth/image_raw', Image, read_depth)
    rospy.Subscriber('/camera/rgb/image_raw', Image, read_rgb)

    # Allow up to one second to connection
    rospy.sleep(1)

if __name__ == '__main__':
    read_image()
