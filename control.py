#!/usr/bin/env python
 
""" timed_out_and_back.py - Version 1.2 2014-12-14
    A basic demo of the using odometry data to move the robot along
    and out-and-back trajectory.
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.5
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""
 
import rospy
from geometry_msgs.msg import Twist
from math import pi
import math
class Turn():
    def __init__(self, angle):
        # Give the node a name
        #rospy.init_node('turn', anonymous=False)
 
        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        
        # How fast will we update the robot's movement?
        rate = 50
        
        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)
        
        # Set the forward linear speed to 0.2 meters per second 
        linear_speed = 0.2

        # Set the travel distance to 1.0 meters
        goal_distance = 0.2
        
        # How long should it take us to get there?
        linear_duration = goal_distance / linear_speed

        # Set the rotation speed to 1.0 radians per second
        if angle < 0:
            angular_speed = -1.0
        else:
            angular_speed = 1.0
        
        # Set the rotation angle to Pi radians (180 degrees)
        goal_angle = angle
        
        # How long should it take to rotate?
        angular_duration = goal_angle / angular_speed

        move_cmd = Twist()

        # Make sure the initial orientation is straight
        move_cmd.linear.x = 0.0

        ticks = int(linear_duration * rate)
            
        for t in range(ticks):
            self.cmd_vel.publish(move_cmd)
            r.sleep()

        # Stop the robot before the rotation
        move_cmd = Twist()
        self.cmd_vel.publish(move_cmd)
        rospy.sleep(1)  
 
        # Set the angular speed
        move_cmd.angular.z = angular_speed

        # Rotate for a time to go 180 degrees
        ticks = abs(int(goal_angle * rate))
            
        for t in range(ticks):           
            self.cmd_vel.publish(move_cmd)
            r.sleep()

        # Stop the robot before the rotation
        move_cmd = Twist()
        self.cmd_vel.publish(move_cmd)
        rospy.sleep(1)

        move_cmd.linear.x = linear_speed  

        
        self.cmd_vel.publish(move_cmd)
        rospy.sleep(2)

        # stop the robot
        self.cmd_vel.publish(Twist())
            
    def shutdown(self):
        # Always stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


class MoveForward():
    def __init__(self):
        # Give the node a name
        #rospy.init_node('move_forward', anonymous=False)
 
        # Set rospy to execute a shutdown function when exiting       
        #rospy.on_shutdown(self.shutdown)
        
        # Publisher to control the robot's speed
        print("Publishing")
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/teleop', Twist, queue_size=1)
        print("Finish publishing")
        # How fast will we update the robot's movement?
        rate = 50
        
        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)
        
        # Set the forward linear speed to 0.2 meters per second 
        linear_speed = 0.2

        move_cmd = Twist()
            
        move_cmd.linear.x = linear_speed
            
        self.cmd_vel.publish(move_cmd)
        rospy.sleep(2)

        # stop the robot
        self.cmd_vel.publish(Twist())
            
    def shutdown(self):
        # Always stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

 
class OutAndBack():
    def __init__(self):
        # Give the node a name
        rospy.init_node('out_and_back', anonymous=False)
 
        # Set rospy to execute a shutdown function when exiting       
        rospy.on_shutdown(self.shutdown)
        
        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # How fast will we update the robot's movement?
        rate = 50
        
        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)
        
        # Set the forward linear speed to 0.2 meters per second 
        linear_speed = 1
        
        # Set the travel distance to 1.0 meters
        goal_distance = 1.0
        
        # How long should it take us to get there?
        linear_duration = goal_distance / linear_speed
        
        # Set the rotation speed to 1.0 radians per second
        angular_speed = 1.0
        
        # Set the rotation angle to Pi radians (180 degrees)
        goal_angle = pi
        
        # How long should it take to rotate?
        angular_duration = goal_angle / angular_speed
     
        # Loop through the two legs of the trip  
        for i in range(2):
            # Initialize the movement command
            move_cmd = Twist()
            
            # Set the forward speed
            move_cmd.linear.x = linear_speed
            
            # Move forward for a time to go the desired distance
            ticks = int(linear_duration * rate)
            
            for t in range(ticks):
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            
            # Stop the robot before the rotation
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)
            
            # Now rotate left roughly 180 degrees  
            
            # Set the angular speed
            move_cmd.angular.z = angular_speed
 
            # Rotate for a time to go 180 degrees
            ticks = int(goal_angle * rate)
            
            for t in range(ticks):           
                self.cmd_vel.publish(move_cmd)
                r.sleep()
                
            # Stop the robot before the next leg
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)    
            
        # Stop the robot
        self.cmd_vel.publish(Twist())
        
    def shutdown(self):
        # Always stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
def control_robot(operation):
    if operation == 1:
        print("MoveForward---------")
        MoveForward()
        
    elif operation == 2:
        print("Turn Right pi/2--------")
        Turn(math.pi/2)
        
    elif operation == 3:
        print("Turn Left pi/2------")
        Turn(-math.pi/2)
        
    elif operation == 4:
        print("Turn Right pi/4----------")
        Turn(math.pi/4)
        
    elif operation == 5:
        print("Turn Left pi/4---------")
        Turn(-math.pi/4)
        
#if __name__ == '__main__':
    #try:
        #print("starting")
        #print("Please input what will turtlebot do?: ")
        #print("1. MoveForward, 2. TurnRight 3. TurnLeft")
        #operation = input()
        #if operation == 1:
            #MoveForward()
        #elif operation == 2:
            #angle = input("Please input what angle will turtlebot turn?(from 0 to pi): ")
            #Turn(-angle)
        #elif operation == 3:
            #angle = input("Please input what angle will turtlebot turn?(from 0 to pi): ")
            #Turn(angle)
    #except:
	#rospy.loginfo("Out-and-Back node terminated.")

