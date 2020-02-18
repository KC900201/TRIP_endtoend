# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 2020

@author: atsumilab
@filename: test_2.py
@coding: utf-8
@description: Tutorial on how to spawn actors, control vehicle pyhsics, and apply sensor (camera)
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/,
      https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/?completed=/introduction-self-driving-autonomous-cars-carla-python/
========================
Date          Comment
========================
02182020      First revision 
"""

import sys
import os
import glob
import random
import time
import cv2
import numpy as np

try:
    #sys.path.append(glob.glob(r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\carla_sim\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768

# Create actor list
actor_list = []

# Function to control physics of vehicle (actor)
def vehicle_control(vehicle):
    # Modify vehicle pyhsics parameters
    physics_control = vehicle.get_physics_control()
    
    physics_control.max_rpm = 10000
    physics_control.moi = 1000
    physics_control.use_gear_autobox = True
    physics_control.gear_switch_time = 100
    
    # Modify vehicle wheels physics parameters
    front_right_wheel = carla.WheelPhysicsControl(tire_friction=5.0, damping_rate=2.0, steer_angle=100.0)
    front_left_wheel = carla.WheelPhysicsControl(tire_friction=5.0, damping_rate=2.0, steer_angle=100.0)
    rear_right_wheel = carla.WheelPhysicsControl(tire_friction=10.0, damping_rate=1.5, steer_angle=90.0)
    rear_left_wheel = carla.WheelPhysicsControl(tire_friction=10.0, damping_rate=1.5, steer_angle=90.0)
    wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]
    
    physics_control.wheels = wheels
    
    # Apply modified controls to vehicle
    vehicle.apply_physics_control(physics_control)
    
    return vehicle

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
    # image.save_to_disk('test/%06d.png' % image.frame)
    return i3 / 255.0

def main():
    try:
        # Set up connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
        
        # Retrieve world map
        world = client.load_world('Town03') # load map town 01
        
        # Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        
        # Create actors
        audi_bp = random.choice(blueprint_library.filter('vehicle.audi.*'))
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Set attributes for actors
        audi_bp.set_attribute("color", "0, 0, 128")
        audi_bp.set_attribute("role_name", "test_audi")
        audi_bp.set_attribute("sticky_control", "false")
        
        camera_bp.set_attribute("image_size_x", f"{IMAGE_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IMAGE_HEIGHT}")
        camera_bp.set_attribute("fov", f"100")
#        camera_bp.set_attribute("iso", f"1200")
        
        # Define spawn point (manual) for actor
        transform_2 = carla.Transform(carla.Location(x=2.5, y=0.7))
        transform = random.choice(world.get_map().get_spawn_points())
        
        # Spawn actor
        audi = world.spawn_actor(audi_bp, transform)
        audi = vehicle_control(audi)
        audi.set_autopilot(True)
#        audi.apply_control(carla.VehicleControl(throttle=2.0, steer=1.0))
        actor_list.append(audi)
        # Spawn sensor, attach to vehicle
        camera = world.spawn_actor(camera_bp, transform_2, attach_to=audi)
        camera.listen(lambda image: process_img(image))
        actor_list.append(camera)
        
        # Delay time
        time.sleep(600) # 4 minutes
        
    finally:
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up!")

# Main function
if __name__ == '__main__':
    main()