#!/usr/bin/env python
#encoding: utf8
import rospy, cv2, math                       
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist           
from std_srvs.srv import Trigger  
from std_msgs.msg import UInt16, UInt16MultiArray              
import numpy as np
import sys

speed = 20
rot = 90
lastTime = rospy.Time()
lastTime.secs = 0
lastError = 0

kp = 0.45
kd = kp * 0.65

def make_points(frame, line):

    height, width, _ = frame.shape
        
    slope, intercept = line
        
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down
        
    if slope == 0:
        slope = 0.1
            
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
        
    return [[x1, y1, x2, y2]]

class DetectLane():
    def __init__(self):
        sub = rospy.Subscriber("/cv_camera/image_raw", Image, self.get_image)
        self.pub = rospy.Publisher("lane", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_org = None
        self.servo = rospy.Publisher('/servo', UInt16, queue_size=1)
        self.motor = rospy.Publisher("/motor", UInt16MultiArray, queue_size= 1)

    def monitor(self,rect,org):
        if rect is not None:
            cv2.rectangle(org,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),(0,255,255),4)
       
        self.pub.publish(self.bridge.cv2_to_imgmsg(org, "bgr8"))
   
    def get_image(self,img):
        try:
            self.image_org = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def detect_edges(self):
        if self.image_org is None:
            return None
    
        frame = self.image_org
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #cv2.imshow("HSV",hsv)
        lower_blue = np.array([90, 120, 0], dtype = "uint8")
        upper_blue = np.array([150, 255, 255], dtype="uint8")
        mask = cv2.inRange(hsv,lower_blue,upper_blue)
        #cv2.imshow("mask",mask)
        
        # detect edges
        edges = cv2.Canny(mask, 50, 100)
        #cv2.imshow("edges",edges)
        
        return edges

    def region_of_interest(self):
        edges = self.detect_edges()

        height, width = edges.shape
        mask = np.zeros_like(edges)

        polygon = np.array([[
            (0, height),
            (0,  height/2),
            (width , height/2),
            (width , height),
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        
        cropped_edges = cv2.bitwise_and(edges, mask)
        cv2.imshow("roi",cropped_edges)
        
        return cropped_edges

    def detect_line_segments(self):
        cropped_edges = self.region_of_interest()

        rho = 1  
        theta = np.pi / 180  
        min_threshold = 10  
        
        line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                        np.array([]), minLineLength=5, maxLineGap=150)

        return line_segments

    def average_slope_intercept(self):
        if self.image_org is None:
                return None
        
        frame = self.image_org
        line_segments = self.detect_line_segments()

        lane_lines = []
        
        if line_segments is None:
            print("no line segments detected")
            return lane_lines

        height, width,_ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1/3
        left_region_boundary = width * (1 - boundary)
        right_region_boundary = width * boundary
        
        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    print("skipping vertical lines (slope = infinity")
                    continue
                
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(make_points(frame, right_fit_average))

        return lane_lines

    def display_lines(self, line_color=(0, 255, 0), line_width=6):
        if self.image_org is None:
            return None
    
        frame = self.image_org()
        lines = self.average_slope_intercept()
        line_image = np.zeros_like(frame)
        
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                    
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        
        return line_image

    def get_steering_angle(self):
        lane_lines = self.average_slope_intercept()
        if self.image_org is None:
            return None
    
        frame = self.image_org
        
        height,width,_ = frame.shape
        
        if len(lane_lines) == 2:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            mid = int(width / 2)
            x_offset = (left_x2 + right_x2) / 2 - mid
            y_offset = int(height / 2)
            
        elif len(lane_lines) == 1:
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
            y_offset = int(height / 2)
            
        elif len(lane_lines) == 0:
            x_offset = 0
            y_offset = int(height / 2)
            
        angle_to_mid_radian = math.atan(x_offset / y_offset)
        angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
        steering_angle = angle_to_mid_deg + 90
        
        return steering_angle

    def display_heading_line(self, line_color=(0, 0, 255), line_width=5 ):
        if self.image_org is None:
            return None
    
        org = self.image_org
        frame = self.display_lines()
        steering_angle = self.get_steering_angle()
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape
        
        steering_angle_radian = steering_angle / 180.0 * math.pi
        
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 2)
        
        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
        self.monitor(heading_image,org)  

        return heading_image

    def control(self):
        global lastError, lastTime, rot, kd, kp, speed        
        m = UInt16MultiArray()
        r = UInt16()

        r.data = rot
        self.servo.publish(r)
        steering_angle = self.get_steering_angle()  
        now = rospy.Time.now()
        dt = now.to_sec() - lastTime.to_sec()
        print(dt)
        deviation = steering_angle - 90
        error = abs(deviation)

        if deviation < 5 and deviation > -5:
            deviation = 0
            error = 0
            speed = 18
            r.data = rot
            self.servo.publish(r)
            
        elif deviation >= 5:
            print ('steering_angle:', steering_angle)
            '''if steering_angle > 135:
                steering_angle = 135
            selfservo.ChangeDutyCycle(2+((180 -steering_angle)/18))'''
            
        elif deviation <= -5:
            print('steering_angle:', steering_angle)
            '''if steering_angle < 60:
                steering_angle = 6
            servo.ChangeDutyCycle(2+((180 -steering_angle)/18))'''

        derivative = kd * (error - lastError) / dt
        proportional = kp * error
        PD = int(speed + derivative + proportional)

        spd = abs(PD)
        if spd > 30:
            spd = 30        
        print(spd)  
        m.data[0] = spd
        m.data[1] = spd

        PDrot = int(rot + derivative + proportional)
        if PDrot < 60:
                PDrot = 60
        if PDrot > 135:
                PDrot = 135

        self.motor.publish(m)
        r.data = 180 - PDrot
        self.servo.publish(r)

        lastError = error
        lastTime = rospy.Time.now()

if __name__ == '__main__':
    rospy.init_node('face_to_face')
    fd = DetectLane()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        fd.control()
        rate.sleep()

