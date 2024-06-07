#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist

def img_process(img):
    height, width = img.shape[:2]
    x_margin, y_min, y_max = int(0.4 * width), int(0.5 * height), int(0.6 * height)
    y_middle = int((y_min + y_max) / 2)

    _, binary_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.rectangle(img, (0, y_min), (x_margin, y_max), 125, 3)
    img = cv2.rectangle(img, (width - x_margin, y_min), (width, y_max), 125, 3)

    _, l_x = binary_img[y_min:y_max, 0:x_margin].nonzero()
    _, r_x = binary_img[y_min:y_max, width - x_margin:width].nonzero()

    leftx = np.average(l_x) if len(l_x) > 500 else 0
    rightx = np.average(r_x) + (width - x_margin) if len(r_x) > 500 else width
    midx = int((leftx + rightx) / 2)

    cv2.circle(img, (int(leftx), y_middle), 10, 0, -1)
    cv2.circle(img, (int(rightx), y_middle), 10, 0, -1)
    cv2.circle(img, (int(midx), y_middle), 10, 0, -1)

    steer = np.arctan2(midx - (width / 2), height - y_middle)
    steer = round(steer, 2) # radian
    img = cv2.line(img, (int(width / 2), height), (midx, y_middle), 0, 5)
    img = cv2.putText(img, str(steer), (150, height), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 125, 3)

    return img, steer

class HocarminiController(Node):
    def __init__(self):
        super().__init__('hocarmini_controller')
        self.steer_ = 0.0
        self.speed_ = 50.0
        self.front_distance_ = 50.0
        self.image_subscriber = self.create_subscription(Image, "image", self.image_callback, 10)
        self.lidar_subscriber = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.br = CvBridge()
        self.publisher = self.create_publisher(Twist, 'hocarmini/cmd_vel', 10)
        self.cmd = Twist()
        self.timer_period = 0.25
        self.timer = self.create_timer(self.timer_period, self.hocarmini_control)

    def image_callback(self, img):
        try:
            cv2_frame = self.br.imgmsg_to_cv2(img, 'mono8').astype('uint8')
            processed_img, self.steer_ = img_process(cv2_frame)
            cv2.imshow("camera", processed_img)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error('CvBridge Error: {}'.format(e))

    def lidar_callback(self, msg):
        mean_dist = msg.ranges[360]    
        # print(len(msg.ranges))  # total 720 points 
        if mean_dist != 0.0:
            self.front_distance_ = round(mean_dist, 2)
            #self.get_logger().info('Receiving Lidar data: {:.2f} m'.format(self.front_distance_))

    def hocarmini_control(self):
        if self.front_distance_ < 0.20:
            self.cmd.linear.x = 5.0  # almost stop 
            self.get_logger().info("Obstacle detected at front!")
        else:
            self.cmd.linear.x = self.speed_
        self.cmd.angular.z = self.steer_   # tunning necessary
        self.publisher.publish(self.cmd)
        #self.get_logger().info(f'Speed: {int(self.speed_)}, Steer: {self.steer_}, Distance:{self.front_distance_:.2f}')
        print(f'Speed: {int(self.speed_)}, Steer: {self.steer_}, Distance:{self.front_distance_:.2f}')

def main(args=None):
    rclpy.init(args=args)
    hocarmini = HocarminiController()
    try:
        rclpy.spin(hocarmini)
    except KeyboardInterrupt:
        pass
    finally:
        hocarmini.destroy_node
