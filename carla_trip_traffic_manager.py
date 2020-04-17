# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:33:43 2020

@author: atsumilab
@filename: carla_trip_traffic_manager.py
@coding: utf-8
@description: New evaluation code for testing TRIP risk prediction module on CARLA-based simulated environment. 4 main scenarios are to be tested:
              1. Near miss 
              2. Traffic accident
              3. Pedestrian / vehicle crossing at road intersection
              4. Non-accident (safe) incident
              Recording will done during experiment to record evaluation progress and will be further used to train proposed module
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/,
      https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/?completed=/introduction-self-driving-autonomous-cars-carla-python/
      RGB color setting: https://www.w3schools.com/colors/colors_picker.asp?colorhex=9c7ec9
========================
Date          Comment
========================
04062020      First revision
04072020      Tweak with traffic manager functions:
              - collision and lane change
04082020      Tweak with traffic manager functions:
              - vehicle ignore and traffic ignore              
"""
import sys
import glob
import os
import cv2
import random
import numpy as np
import logging
import time

try:
    #sys.path.append(glob.glob(r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\carla_sim\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IMAGE_WIDTH = 1280 #800 
IMAGE_HEIGHT = 720 #600 

# Create list of actors for test agent, NPC (vehicles, pedestrians)
actor_list = []
vehicle_list = []
walker_list = []
all_id = []

# Process image function
def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
    image.save_to_disk('_out/%08d' % image.frame) # 02282020
    return i3 / 255.0

def process_img_collision(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
    image.save_to_disk('_out/%08d' % image.frame) # 02282020
    return i3 / 255.0

# Start recording function
def start_replay():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
    
        client.set_replayer_time_factor(1.0)
        print(client.replay_file("test_record_01.log", 0.0, 120.0, 0))
    
    finally:
        pass  
    
# Main method
def main():
    try:
        # 1. Set up CARLA client connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        # 3. Retrieve world from CARLA simulation
        world = client.get_world()
#        world = client.load_world("Town01")
#        world = client.load_world(random.choice(client.get_available_maps()).split("/")[4])
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        # 3.1 Create traffic manager client
        tm = client.get_trafficmanager() # default port = 8000
        
        # 4. Create test agents
        test_agent_bp = blueprint_library.filter('vehicle.*')
        test_agent_bp = random.choice([x for x in test_agent_bp if int(x.get_attribute('number_of_wheels')) == 4])
        test_cam_bp = blueprint_library.find('sensor.camera.rgb')
#        test_cam_collision_bp = blueprint_library.find('sensor.camera.rgb')
        # Create test vehicles for collision detection (traffic manager) - 04072020
#        test_collision_bp = blueprint_library.filter('vehicle.tesla.*') 
#        test_collision_bp = random.choice([x for x in test_collision_bp if int(x.get_attribute('number_of_wheels')) == 4])        
        # 4.1 Print test agents blueprints
        print(test_agent_bp)
#        print(test_collision_bp)

        # Start recording process
        # Log will automatically be saved to CarlaUE4\Saved folder as 
        # 'filename.log' if no specific folder is mentioned in parameter
        print("Recording on file: %s" % client.start_recorder("test_record_01.log"))

        # 5. Set attribute for test agents
        test_agent_bp.set_attribute('color', '204, 0, 255') # set color as purple
        test_agent_bp.set_attribute('role_name', 'autopilot')
        test_cam_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
        test_cam_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        test_cam_bp.set_attribute("fov", f"110")
 #       test_collision_bp.set_attribute('role_name', 'autopilot')
 #       test_cam_collision_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
 #       test_cam_collision_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
 #       test_cam_collision_bp.set_attribute("fov", f"110")

        world.wait_for_tick()
        
        # 7. Define spawn point (manual) for test agents
        transform = random.choice(world.get_map().get_spawn_points())
#        transform = carla.Transform(carla.Location(x=3.0, y=1.1, z=0.7))
        transform_2 = carla.Transform(carla.Location(x=2.5, y=1.1, z=0.7))
#        transform_3 = random.choice(world.get_map().get_spawn_points())
#        transform_3 = carla.Transform(carla.Location(x=4.0, y=1.1, z=0.7))
#        transform_4 = carla.Transform(carla.Location(x=2.5, y=1.2, z=0.8))
        # 7.1 Spawn test agents to simulation
        test_agent = world.try_spawn_actor(test_agent_bp, transform)
        test_agent.set_autopilot(True)
        test_agent.apply_control(carla.VehicleControl(gear=3, throttle=2.0, steer=1.0, hand_brake=False))
        # 7.1.1 Spawn collision vehicle to simulation
#        test_collision = world.try_spawn_actor(test_collision_bp, transform_3)
#        test_collision.set_autopilot(True)
        # 7.1.2 Setting Traffic Manager to test agent
#        tm.ignore_lights_percentage(test_collision, 90)
#        tm.distance_to_leading_vehicle(test_collision, 0)
#        tm.vehicle_percentage_speed_difference(test_collision, -30)
#        tm.collision_detection(test_collision, test_agent, True)
#        tm.collision_detection(test_agent, test_collision, True)
        tm.distance_to_leading_vehicle(test_agent, 1)
        tm.ignore_walkers_percentage(test_agent, 80)
        tm.ignore_vehicles_percentage(test_agent, 90)
        tm.ignore_lights_percentage(test_agent, 90)
        tm.vehicle_percentage_speed_difference(test_agent, -20)
#        tm.auto_lane_change(test_agent, True)
#        actor_list.append(test_collision)
        actor_list.append(test_agent)
        # 7.2 Spawn sensor, attach to test vehicle
        test_cam = world.try_spawn_actor(test_cam_bp, transform_2, attach_to=test_agent)
        test_cam.listen(lambda image: process_img(image))
#        test_cam_collision = world.try_spawn_actor(test_cam_collision_bp, transform_4, attach_to=test_collision)
#        test_cam_collision.listen(lambda image: process_img_collision(image))
        actor_list.append(test_cam) 
#        actor_list.append(test_cam_collision)
        
        print('spawned %d test agents' % len(actor_list))

        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)

    finally:
        print("\nDestroying %d actors" % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])
        # reset traffic lights for traffic manager
        tm.reset_traffic_lights()
        # Stop recording
        print("Stop recording")
        client.stop_recorder()
        print("\nAll cleaned up.")
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")
