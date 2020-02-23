# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 2020

@author: atsumilab
@filename: test_3.py
@coding: utf-8
@description: Tutorial on how to spawn actors, control vehicle pyhsics, and apply sensor (camera)
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/,
      https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/?completed=/introduction-self-driving-autonomous-cars-carla-python/
========================
Date          Comment
========================
02232020      First revision 
02232020      Spawn pedestrians, apply walker control
"""

import sys
import os
import glob
import random
import logging
import cv2
import time
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

actor_list = []

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

# Camera function
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
    # image.save_to_disk('test/%06d.png' % image.frame)
    return i3 / 255.0

# Function to spawn pedestrian(world) - 02/23/2020
def spawn_walker(world):
    
    blueprint_library = world.get_blueprint_library()
    walker_bp = blueprint_library.filter('walker.*')
     
    # Spawn points 
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    number_of_npc = random.randint(0, 100)
     
    print("Number of spawn points: %d" % int(number_of_spawn_points))
    print("Number of NPC: %d" % int(number_of_npc))

    if number_of_npc <= number_of_spawn_points:        
        random.shuffle(spawn_points)
    else:
        number_of_npc = number_of_spawn_points      
        
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_npc:
            break
        blueprint = random.choice(walker_bp)
#        if blueprint.has_attribute('age'):
#            age = random.choice(blueprint.get_attribute('age').recommended_values)
#            blueprint.set_attribute('age', age)
#        if blueprint.has_attribute('gender'):
#            gender = random.choice(blueprint.get_attribute('gender').recommended_values)
#            blueprint.set_attribute('gender', gender)
        batch.append(carla.command.SpawnActor(blueprint, transform))
        
    return batch

def main():
    try:
        # Set up connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0) # wait connection setup for 3 minutes
        
        # retrieve world map
        world = client.get_world()
        
        # retrieve blueprint library
        blueprint_library = world.get_blueprint_library()
        
        # Print blueprint library
        '''
        blueprints = [bp for bp in world.get_blueprint_library().filter('walker.*')]
        for blueprint in blueprints:
            print('Every bp id: {}'.format(blueprint.id))
            for attr in blueprint:
                print(' - {}'.format(attr))
        '''
        
        # Create camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        walker_bp = blueprint_library.find('walker.pedestrian.0001')
        
        # Set attributes for actors
        # walker_bp.set_attribute("gender", "male")

        camera_bp.set_attribute("image_size_x", f"{IMAGE_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IMAGE_HEIGHT}")
        camera_bp.set_attribute("fov", f"100")

        # Define spawn point (manual) for actor
        transform = carla.Transform(carla.Location(x=2.0, y=5.0))
        transform2 = random.choice(world.get_map().get_spawn_points())
        
        # Spawn actor
        walker_test = world.try_spawn_actor(walker_bp, transform2)
        
        walker_test.apply_control(carla.WalkerControl(carla.Vector3D(x=1.0, y=1.0, z=1.0), speed=1.0, jump=False))
        actor_list.append(walker_test)
        
        camera = world.try_spawn_actor(camera_bp, transform, attach_to=walker_test)
        camera.listen(lambda image: process_img(image))
        actor_list.append(camera)
        
        walker_batch = spawn_walker(world)
        
        for response in client.apply_batch_sync(walker_batch):
            if response.error:
                logging.error(response.error)
            else:
                actor_list.append(response.actor_id)
    
        for actor in actor_list:
            print(actor.attributes)
            if "walker" in actor.type_id:
                actor.apply_control(carla.WalkerControl(speed=1.0, jump=False))
    
        print('spawned %d NPC actors' % len(actor_list))
        
        while True:
            world.wait_for_tick() # wait for world to get actors
        
    finally:
        print("\n Destroying %d actors" % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("All cleaned up!")
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
