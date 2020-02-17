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
"""

import os
import glob
import sys
import random

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

try:
    sys.path.append(glob.glob(r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\carla_sim\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

actor_list = []

def main():
#    print("Testing")
    try:
        # Connect to Carla client
        client = carla.Client('localhost', 2000) # localhost, TCP port 2000 default
        client.set_timeout(5.0)
    
        # Print retrieve available maps
        print(client.get_available_maps())

        # retrieve world - random retrieve map town 07
        world = client.load_world('Town07')
    
        # Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
    
        # Random retrieve vehicle and camera
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        camera_bp = random.choice(blueprint_library.filter('sensor.camera.*'))
        print(vehicle_bp)
        print(camera_bp)

        # Setting attributes
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '0, 255, 0')
        if camera_bp.has_attribute('image_size_x'):
            camera_bp.set_attribute('image_size_x', IMAGE_WIDTH)
        if camera_bp.has_attribute('image_size_y'):
            camera_bp.set_attribute('image_size_y', IMAGE_HEIGHT)

        # Find spawn points for actors
        spawn_point = random.choince(world.get_map().get_spawn_points())

        # Spawn actors in points
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        camera = world.spawn_actor(camera_bp, spawn_point,attach_to=vehicle_bp) # spawn camera, attach to vehicle
        # Append spawn actors to list
        actor_list.append(vehicle)
        actor_list.append(camera)

        # Continue - 02172020 (midnight)
    
    finally: 
        for actor in actor_list:
            actor.destroy()
        print("All cleaned up")

# Main function
if __name__ == '__main__':
    main()
