# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 2020

@author: atsumilab
@filename: trial_1.py
@coding: utf-8
@description: First attempt to develop a CARLA simulation environment for TRIP evaluation
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/,
      https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/?completed=/introduction-self-driving-autonomous-cars-carla-python/
========================
Date          Comment
========================
02172020      First revision 
02182020      Apply vehicle control, spawn vehicle and weather control
"""

import os
import glob
import sys
import random
import time
import numpy as np
import cv2

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 768

try:
    #sys.path.append(glob.glob(r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\carla_sim\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []

# Process image
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(5) # delay 5 seconds
    return i3 / 255.0

def main():
#    print("Testing")
    try:
        # Connect to Carla client
        client = carla.Client('localhost', 2000) # localhost, TCP port 2000 default
        client.set_timeout(5.0)
    
        # retrieve world - random retrieve map town 07
        #world = client.load_world('Town06')
        world = client.get_world()

        # Print retrieve available maps
        #print(client.get_available_maps())
    
        # Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
    
        # Random retrieve vehicle and camera
        #vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.cybertruck'))
        #camera_bp = random.choice(blueprint_library.filter('sensor.camera.rgb'))
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        collision_bp = blueprint_library.find('sensor.other.collision')
        #print(vehicle_bp)
        #print(camera_bp)

        # Setting attributes
        #if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute('color', '0, 255, 0')
        vehicle_bp.set_attribute('role_name', 'Test_Carla_TRIP')
        vehicle_bp.set_attribute('sticky_control', 'true')        
    
        if camera_bp.has_attribute('image_size_x'):
            camera_bp.set_attribute('image_size_x', f"{IMAGE_WIDTH}")
        if camera_bp.has_attribute('image_size_y'):
            camera_bp.set_attribute('image_size_y', f"{IMAGE_HEIGHT}")
        camera_bp.set_attribute('fov', f'110')
        
        collision_bp.set_attribute('role_name', 'test_collision')
        
        # Find spawn points for actors
        spawn_point = random.choice(world.get_map().get_spawn_points())

        # Spawn actors in points
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Set vehicle to move autopilot - 02182020
        vehicle.set_autopilot(True)
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        
        camera = world.spawn_actor(camera_bp, spawn_point,attach_to=vehicle) # spawn camera, attach to vehicle
        camera.listen(lambda image: process_img(image))
        
        collision_sensor = world.spawn_actor(collision_bp, spawn_point, attach_to=vehicle) # spawn collision sensor, attach to vehicle
        # Append spawn actors to list
        actor_list.append(vehicle)
        actor_list.append(camera)
        actor_list.append(collision_sensor)
        # Start recording, save recording
        client.start_recorder("test_record01.log")
    
        time.sleep(180) # delay 180 seconds
        
        # Stop recording
        client.stop_recorder()
        client.replay_file("test_record01.log")
    
    finally: 
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up")

# Main function
if __name__ == '__main__':
    main()
