# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:59:26 2020

@author: KwongCheongNg, atsumilab
@filename: carla_trip_evaluation.py
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
02262020      First revision 
02282020      Fix saving image (images.save_to_disk) - original format,
              Fix NPC spawn amount <= total amt of available spawn points
02292020      Separate spawn NPC and spawn test agent functions (spawn_NPC)
03092020      Apply TRIP risk prediction function in CARLA sensor
03132020      Apply TRIP dataset generation function in CARLA sensor
03142020      Create diverse spawn NPC functions according to vehicle class: car, motorbike, walker, bicycle
03202020      Code modification in dataset generation for TRIP module
03282020      Include video risk prediction for TRIP module during CARLA evaluation process
04162020      Implement Traffic Manager module during spawning AI and test agent (CARLA v0.9.8)
04172020      Implement Traffic Manager module to every spawn function
04252020      New function to spawn test agent and AI and implement Traffic Manager
05082020      Separate vehicle list into 3 individual lists according to vehicle type (4 wheels, 2 wheels - bikes, bicycle),
              Implement collision detector in test agent
05132020      Implement checking of None object type if spawning actors and test agents
"""
import sys
import glob
import os
import cv2
import random
import numpy as np
import logging
import time
import argparse

try:
    sys.path.append(glob.glob(r'CARLA_0.9.8_project/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % ( # 04162020
#    sys.path.append(glob.glob(r'CARLA_0.9.7_project/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (        
#    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Import TRIP module - video prediction
#from risk_prediction.trip_vpredictor_carla import TripVPredictorCarla 
#from risk_prediction.trip_predictor_carla import TripPredictorCarla
from estimation.dataset_generator.dataset_generator_function import DatasetGenerator # 03132020
from estimation.dataset_generator.object_detector import ObjectDetector # 03132020
from risk_prediction.trip_vpredictor import TripVPredictor

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Create list of actors for test agent, NPC (vehicles, pedestrians)
actor_list = []
vehicle_list = []
walker_list = []
all_id = []
# 05082020
car_list = []
bike_list = []

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

# 03132020   
def predict_risk_img(image, output_dir):
    # Process image
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
#    image.save_to_disk(output_dir + '/test/orig_img/%08d' % image.frame)
#    image.save_to_disk(output_dir + '/Town01/orig_img/%08d' % image.frame)
    image.save_to_disk(output_dir + '/orig_img/%08d' % image.frame)
    return i3 / 255.0

# Start recording function
def start_replay():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    try:
    
        client.set_replayer_time_factor(1.0)
        print(client.replay_file("test_record_01.log", 0.0, 120.0, 0))
    
    finally:
        pass    
 
# Spawn Pedestrian - 03142020
def spawn_walker():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    try:
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        
        # 3. Retrieve world from CARLA simulation        
#        world = client.get_world()
        world = client.load_world('Town04')
        print(world.get_map().name)
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        npc_walker_bp = blueprint_library.filter('walker.pedestrian.*')
        # ----------------------
        # 4 Spawn NPC walkers    
        # ----------------------
        # some settings
        percentagePedestriansRunning = 1.0      # how many pedestrians will run
        percentagePedestriansCrossing = 30.0     # how many pedestrians will walk through the road
        # 4.1 take all random locations to spawn
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        npc_walker_amt = random.randint(150, num_spawn_points) 
        
        if npc_walker_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_walker_amt > num_spawn_points:
            msg = 'Requested %d walkers, but could only find %d spawn points'
            logging.warning(msg, npc_walker_amt, num_spawn_points)
            print('Requested %d walkers, but could only find %d spawn points' % (npc_walker_amt, num_spawn_points))
            npc_walker_amt = int(num_spawn_points / 2)

        spawn_points = []
        for i in range(npc_walker_amt):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 6.2.2 spawn walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(npc_walker_bp)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set max speed
            if walker_bp.has_attribute('speed'):
                if(random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 6.2.3 spawn walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walker_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list[i]["con"] = results[i].actor_id
        # 6.2.4 we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walker_list)):
            all_id.append(walker_list[i]["con"])
            all_id.append(walker_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # 6.2.5 initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print('Spawned %d walkers, press Ctrl+C to exit.' % len(walker_list))
        
        world.wait_for_tick()

        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)

    finally:        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        
        print("\nDestroying %d walkers" % len(walker_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in all_id])

# Spawn Car - 03142020
def spawn_car():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    try:
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        # 3. Retrieve world from CARLA simulation        
#        world = client.load_world('Town01')
        world = client.get_world()
        print(world.get_map().name)
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        # 3.1 Create traffic manager client
        tm = client.get_trafficmanager() # default port = 8000

        # 4 Setting synchronous mode (04172020)
#        synchronous_master = False
#        settings = world.get_settings()
#        tm.set_synchronous_mode(True)
        '''
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False            
        '''
        # 6 Spawn NPC agents in environment
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')
        # Sort out blueprint (car)
        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('carlacola')]
        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        npc_amt = random.randint(100, num_spawn_points) # 02282020 #03022020 (minimum no of NPC vehicles = 100)

        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error

        for n, transform in enumerate(spawn_points):
            if n >= npc_amt:
                break
            blueprint = random.choice(npc_vehicle_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            vehicle.set_autopilot(enabled=True)
            # 04172020
            tm.ignore_lights_percentage(vehicle, 90) # 04062020
            tm.distance_to_leading_vehicle(vehicle, 30)
            tm.ignore_walkers_percentage(vehicle, 30)
            tm.ignore_vehicles_percentage(vehicle, 90)
            tm.auto_lane_change(vehicle, False)
            vehicle_list.append(vehicle)            

        print('Spawned %d vehicles, press Ctrl+C to exit.' % len(vehicle_list))
        
        # 7 set global speed difference for Traffic Manager
#        tm.global_percentage_speed_difference(30.0)
        
#        world.wait_for_tick()

        while True: # wait for world to get actors
            world.wait_for_tick()
            time.sleep(1)

    finally:
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])

def spawn_motorbike():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)

    try:        
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        # 3. Retrieve world from CARLA simulation
        world = client.get_world()
        print(world.get_map().name)
        # 3.1 Create traffic manager client
        tm = client.get_trafficmanager() # default port = 8000
        
        # 4 Setting synchronous mode (04172020)
        synchronous_master = False
        settings = world.get_settings()
        tm.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False
        
        # 6 Spawn NPC agents in environment
        blueprint_library = world.get_blueprint_library()
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')

        # Avoid spawning NPC prone to accident
        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 2]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('bike')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('century')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('omafiets')]

        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        #npc_amt = NPC_AMT
        npc_amt = random.randint(150, num_spawn_points) # 02282020 #03022020 (minimum no of NPC vehicles = 100)
        
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error
        
        for n, transform in enumerate(spawn_points):
            if n >= npc_amt:
                break
            blueprint = random.choice(npc_vehicle_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            vehicle.set_autopilot(True)
            # 04172020
            tm.ignore_lights_percentage(vehicle, 90) # 04062020
            tm.distance_to_leading_vehicle(vehicle, 30)
            tm.ignore_walkers_percentage(vehicle, 80)
            tm.ignore_vehicles_percentage(vehicle, 80)
            tm.auto_lane_change(vehicle, False)
            vehicle_list.append(vehicle)

        print('Spawned %d vehicles, press Ctrl+C to exit.' % len(vehicle_list))
        
#        world.wait_for_tick()

        while True: # wait for world to get actors
            if synchronous_master:
                world.tick() 
            else:
                world.wait_for_tick()
                time.sleep(1)

    finally:
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])

def spawn_bicycle():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    try:
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        # 3. Retrieve world from CARLA simulation
        world = client.get_world()
        print(world.get_map().name)
        # 3.1 Create traffic manager client
        tm = client.get_trafficmanager() # default port = 8000

        # 4 Setting synchronous mode (04172020)
        synchronous_master = False
        settings = world.get_settings()
        tm.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False            

        # 6 Spawn NPC agents in environment
        blueprint_library = world.get_blueprint_library()
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')

        # Avoid spawning NPC prone to accident
        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 2]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if x.id.endswith('bike') or x.id.endswith('bike') or x.id.endswith('omafiets')]
#        npc_vehicle_bp = [x for x in npc_vehicle_bp if x.id.endswith('century')]
#        npc_vehicle_bp = [x for x in npc_vehicle_bp if x.id.endswith('omafiets')]

        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        #npc_amt = NPC_AMT
        npc_amt = random.randint(150, num_spawn_points) # 02282020 #03022020 (minimum no of NPC vehicles = 100)
        
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error
        
        for n, transform in enumerate(spawn_points):
            if n >= npc_amt:
                break
            blueprint = random.choice(npc_vehicle_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            vehicle.set_autopilot(enabled=True)
            # 04172020
            tm.ignore_lights_percentage(vehicle, 90) # 04062020
            tm.distance_to_leading_vehicle(vehicle, 30)
            tm.ignore_walkers_percentage(vehicle, 80)
            tm.ignore_vehicles_percentage(vehicle, 80)
            tm.auto_lane_change(vehicle, False)
            vehicle_list.append(vehicle)

        print('Spawned %d vehicles, press Ctrl+C to exit.' % len(vehicle_list))
        
        # 7 set global speed difference for Traffic Manager
        tm.global_percentage_speed_difference(30.0)

#        world.wait_for_tick()

        while True: # wait for world to get actors
            if synchronous_master:
                world.tick() 
            else:
                world.wait_for_tick()
                time.sleep(1)
           
    finally:
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])

def spawn_lorry():
    try:
        # 1. Set up CARLA client connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    finally:
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])

# Spawn NPC function - 02292020
def spawn_npc():
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)

    try:
        # 1. Set up CARLA client connection
        # 1.1. Set up Traffic Manager - 04062020
        traffic_manager = client.get_trafficmanager()
        traffic_manager.global_percentage_speed_difference(25.0)
        
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        
        # 3. Retrieve world from CARLA simulation        
#        world = client.load_world(random.choice(client.get_available_maps()).split("/")[4])
#        world = client.load_world("Town04")
        world = client.get_world()
        print(world.get_map().name)
#        print(world)
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()

        # 6 Spawn NPC agents in environment
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')
        npc_walker_bp = blueprint_library.filter('walker.pedestrian.*')
        # Avoid spawning NPC prone to accident
        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('isetta')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('carlacola')]
        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        #npc_amt = NPC_AMT
        npc_amt = random.randint(100, num_spawn_points) # 02282020 #03022020 (minimum no of NPC vehicles = 100)
        
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error
        
        for n, transform in enumerate(spawn_points):
            if n >= npc_amt:
                break
            blueprint = random.choice(npc_vehicle_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            #vehicle.set_autopilot(True)
            vehicle.set_autopilot(enabled=True)
            traffic_manager.ignore_lights_percentage(vehicle, 70) # 04062020
#            traffic_manager.vehicle_percentage_speed_difference(vehicle, -20) # 04062020
#            traffic_manager.distance_to_leading_vehicle(vehicle, 30)
            traffic_manager.ignore_walkers_percentage(vehicle, 40)
            traffic_manager.ignore_vehicles_percentage(vehicle, 88)
#            traffic_manager.auto_lane_change(vehicle, False)
            vehicle_list.append(vehicle)            
        # ----------------------
        # 6.2 Spawn NPC walkers    
        # ----------------------
        # some settings
        percentagePedestriansRunning = 10.0      # how many pedestrians will run
        percentagePedestriansCrossing = 60.0     # how many pedestrians will walk through the road
        # 6.2.1 take all random locations to spawn
        spawn_points = []
        #npc_walker_amt = NPC_WALKER_AMT
        npc_walker_amt = random.randint(50, npc_amt) # 02282020 #03022020 (minimum number of NPC walker = 50)
        
        if npc_walker_amt > npc_amt:
            npc_walker_amt = npc_amt
            npc_walker_amt = int(npc_walker_amt / 2)
        else:
            npc_walker_amt = npc_walker_amt
        
        for i in range(npc_walker_amt):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 6.2.2 spawn walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(npc_walker_bp)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set max speed
            if walker_bp.has_attribute('speed'):
                if(random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 6.2.3 spawn walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walker_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list[i]["con"] = results[i].actor_id
        # 6.2.4 we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walker_list)):
            all_id.append(walker_list[i]["con"])
            all_id.append(walker_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # 6.2.5 initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print('Spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicle_list), len(walker_list)))
        
        world.wait_for_tick()

        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)

    finally:
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])
        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        
        print("\nDestroying %d walkers" % len(walker_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in all_id])

# 05082020
def main_traffic_manager_2():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    tm = client.get_trafficmanager() # default port = 8000
#    world = client.load_world("Town05")
    try:
        # Create output directory for img capture
        parser = argparse.ArgumentParser(description='dataset_maker')
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2\training\Town05\Phase 1', help='directory where the dataset will be created')
        parser.add_argument('--output_dir', default=r'C:\Users\user\Pictures\CARLA_dataset\test_2\training\Town05\Phase 1', help='directory where the dataset will be created')
        args = parser.parse_args()
        output_dir = args.output_dir
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        # 2.1
        world = client.get_world()
        print(world.get_map().name)
        blueprint_library = world.get_blueprint_library()
        # 4. Create test agents
        test_agent_bp = blueprint_library.filter('vehicle.*')
        test_agent_bp = random.choice([x for x in test_agent_bp if int(x.get_attribute('number_of_wheels')) == 4 and x is not None])
        test_cam_bp = blueprint_library.find('sensor.camera.rgb')
        test_collision_bp = blueprint_library.find('sensor.other.collision')
        test_li_bp = blueprint_library.find('sensor.other.lane_invasion')
        # 4.1 Print test agents blueprints
        print(test_agent_bp)
        print(test_cam_bp)

        # Start recording process
        # Log will automatically be saved to CarlaUE4\Saved folder as 
        # 'filename.log' if no specific folder is mentioned in parameter
        print("Recording on file: %s" % client.start_recorder("test_record_01.log"))

        # 5. Set attribute for test agents
        test_agent_bp.set_attribute('color', '204, 0, 255') # set color as purple
        test_agent_bp.set_attribute('role_name', 'autopilot')
        test_cam_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
        test_cam_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        test_cam_bp.set_attribute("fov", f"100")
        
        world.wait_for_tick()
        # 6 Spawn NPC agents in environment
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')
        npc_walker_bp = blueprint_library.filter('walker.pedestrian.*')
        # Avoid spawning NPC prone to accident
#        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('isetta')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('carlacola')]
        # Categorize vehicle bp to 4 wheels and 2 wheels
#        print(npc_car_bp.get_attribute('id'))
#        print(npc_bike_bp)
        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        print("Number of spawn points: %d" % int(num_spawn_points))
        npc_amt = int(num_spawn_points / 2)
        
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error

        npc_car_amt = int(npc_amt / 2) 
        npc_bike_amt =  npc_amt - npc_car_amt
        #---------------------
        # 6.1.1 Spawn NPC car
        #---------------------
        for n, transform in enumerate(spawn_points):
            if n >= npc_car_amt:
                break
            blueprint = random.choice([x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4 and x is not None])
#            print(blueprint)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            car = world.try_spawn_actor(blueprint, transform)
            if(not isinstance(car, type(None))): # 05132020
                car.set_autopilot(enabled=True)
#            tm.ignore_lights_percentage(car, 90) # 04062020
#            traffic_manager.vehicle_percentage_speed_difference(car, -20) # 04062020
#            traffic_manager.distance_to_leading_vehicle(car, 30)
#            tm.ignore_walkers_percentage(car, 80)
#            tm.ignore_vehicles_percentage(car, 90)
#            traffic_manager.auto_lane_change(car, False)
                car_list.append(car)    
        #------------------------
        # 6.1.2 Spawn NPC bicycle
        #------------------------
        for n, transform in enumerate(spawn_points):
            if n >= npc_bike_amt:
                break
            blueprint = random.choice([x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 2 and x is not None])
            print(blueprint)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            bike = world.try_spawn_actor(blueprint, transform)
            if(not isinstance(bike, type(None))): # 05132020
#                print(type(bike))
                bike.set_autopilot(enabled=True)
#            tm.ignore_lights_percentage(bike, 90) # 04062020
#            traffic_manager.vehicle_percentage_speed_difference(bike, -20) # 04062020
#            traffic_manager.distance_to_leading_vehicle(bike, 30)
#            tm.ignore_walkers_percentage(bike, 80)
#            tm.ignore_vehicles_percentage(bike, 90)
#            traffic_manager.auto_lane_change(bike, False)
                bike_list.append(bike)
        # ----------------------
        # 6.2 Spawn NPC walkers    
        # ----------------------
        # some settings
        percentagePedestriansRunning = 40.0      # how many pedestrians will run
        percentagePedestriansCrossing = 70.0     # how many pedestrians will walk through the road
        # 6.2.1 take all random locations to spawn
        spawn_points = []
        #npc_walker_amt = NPC_WALKER_AMT
        #npc_walker_amt = random.randint(100, npc_amt) # 02282020 #03022020 (minimum number of NPC walker = 50)
        npc_walker_amt = int(num_spawn_points / 2)
        '''
        if npc_walker_amt > npc_amt:
            npc_walker_amt = npc_amt
            npc_walker_amt = int(npc_walker_amt / 2)
        else:
            npc_walker_amt = npc_walker_amt
        '''
        for i in range(npc_walker_amt):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 6.2.2 spawn walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(npc_walker_bp)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set max speed
            if walker_bp.has_attribute('speed'):
                if(random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 6.2.3 spawn walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walker_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list[i]["con"] = results[i].actor_id
        # 6.2.4 we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walker_list)):
            all_id.append(walker_list[i]["con"])
            all_id.append(walker_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # 6.2.5 initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
            
        print('Spawned %d cars, %d bikes and %d walkers.' % (len(car_list), len(bike_list), len(walker_list)))
        
        # 7. Define spawn point (manual) for test agents
        transform = random.choice(world.get_map().get_spawn_points())
        transform_2 = carla.Transform(carla.Location(x=2.5, y=1.1, z=0.7))
        transform_3 = carla.Transform(carla.Location(x=2.6, y=1.1, z=0.7))
        # 7.1 Spawn test agents to simulation
        test_agent = world.try_spawn_actor(test_agent_bp, transform)
        if(not isinstance(test_agent, type(None))): # 05132020
            test_agent.set_autopilot(enabled=True)
            test_agent.apply_control(carla.VehicleControl(gear=1, throttle=1.0, steer=0.5, hand_brake=False))
    #        test_agent.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, hand_brake=False))
            # 7.1.1 Setting Traffic Manager to test agent (04162020)
            tm.distance_to_leading_vehicle(test_agent, 1)
            tm.ignore_walkers_percentage(test_agent, 80)
            tm.ignore_vehicles_percentage(test_agent, 90)
            tm.ignore_lights_percentage(test_agent, 90)
    #        tm.auto_lane_change(test_agent, True)
            tm.force_lane_change(test_agent, True)
    #        tm.vehicle_percentage_speed_difference(test_agent, -20)
            actor_list.append(test_agent)
        # 7.2 Spawn sensor, attach to test vehicle
        test_cam = world.try_spawn_actor(test_cam_bp, transform_2, attach_to=test_agent)
        test_collision = world.try_spawn_actor(test_collision_bp, transform_3, attach_to=test_agent)
        if(not (isinstance(test_cam, type(None)) or isinstance(test_collision, type(None)))): # 05132020
#        test_cam.listen(lambda image: process_img(image))
            test_cam.listen(lambda image: predict_risk_img(image, output_dir))
            actor_list.append(test_cam)
            actor_list.append(test_collision)

        print('spawned %d test agents, press Ctrl+C to exit.' % len(actor_list))

        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)
        
    finally:
        print("\nDestroying %d actors" % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])
        print("\nDestroying %d cars" % len(car_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in car_list])        
        print("\nDestroying %d bikes" % len(bike_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in bike_list])        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        
        print("\nDestroying %d walkers" % len(walker_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in all_id])        
        print("\n Reset traffic lights")
        tm.reset_traffic_lights()
        # Stop recording
        print("\nStop recording")
        client.stop_recorder()
#        generate_data()
        print("\nAll cleaned up.")

# 04252020
def main_traffic_manager():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    # 1.1 Create traffic manager client
    tm = client.get_trafficmanager() # default port = 8000
    world = client.load_world("Town04")
    try:
        parser = argparse.ArgumentParser(description='dataset_maker')
        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2\training\Town04\Phase 3', help='directory where the dataset will be created')
        args = parser.parse_args()
        output_dir = args.output_dir
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        # 2.1
        world = client.get_world()
        print(world.get_map().name)
        blueprint_library = world.get_blueprint_library()
        # 3.3 Setting synchronous mode (04172020)
#        synchronous_master = False
#        settings = world.get_settings()
#        tm.set_synchronous_mode(True)
        # 4. Create test agents
        test_agent_bp = blueprint_library.filter('vehicle.*')
        test_agent_bp = random.choice([x for x in test_agent_bp if int(x.get_attribute('number_of_wheels')) == 4])
        test_cam_bp = blueprint_library.find('sensor.camera.rgb')
        # 4.1 Print test agents blueprints
        print(test_agent_bp)
        print(test_cam_bp)

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
                        
        world.wait_for_tick()
        
        # 6 Spawn NPC agents in environment
        npc_vehicle_bp = blueprint_library.filter('vehicle.*')
        npc_walker_bp = blueprint_library.filter('walker.pedestrian.*')
        # Avoid spawning NPC prone to accident
        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('isetta')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('carlacola')]
        # ---------------------
        # 6.1 Spawn NPC vehicle    
        # ---------------------
        spawn_points = world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        #npc_amt = NPC_AMT
        npc_amt = random.randint(150, num_spawn_points) # 02282020 #03022020 (minimum no of NPC vehicles = 100)
        
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error
        
        for n, transform in enumerate(spawn_points):
            if n >= npc_amt:
                break
            blueprint = random.choice(npc_vehicle_bp)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle = world.try_spawn_actor(blueprint, transform)
            #vehicle.set_autopilot(True)
            vehicle.set_autopilot(enabled=True)
#            tm.ignore_lights_percentage(vehicle, 90) # 04062020
#            traffic_manager.vehicle_percentage_speed_difference(vehicle, -20) # 04062020
#            traffic_manager.distance_to_leading_vehicle(vehicle, 30)
#            tm.ignore_walkers_percentage(vehicle, 80)
#            tm.ignore_vehicles_percentage(vehicle, 90)
#            traffic_manager.auto_lane_change(vehicle, False)
            vehicle_list.append(vehicle)            
        # ----------------------
        # 6.2 Spawn NPC walkers    
        # ----------------------
        # some settings
        percentagePedestriansRunning = 40.0      # how many pedestrians will run
        percentagePedestriansCrossing = 70.0     # how many pedestrians will walk through the road
        # 6.2.1 take all random locations to spawn
        spawn_points = []
        #npc_walker_amt = NPC_WALKER_AMT
        npc_walker_amt = random.randint(100, npc_amt) # 02282020 #03022020 (minimum number of NPC walker = 50)
        '''
        if npc_walker_amt > npc_amt:
            npc_walker_amt = npc_amt
            npc_walker_amt = int(npc_walker_amt / 2)
        else:
            npc_walker_amt = npc_walker_amt
        '''
        for i in range(npc_walker_amt):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 6.2.2 spawn walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(npc_walker_bp)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set max speed
            if walker_bp.has_attribute('speed'):
                if(random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 6.2.3 spawn walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walker_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walker_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walker_list[i]["con"] = results[i].actor_id
        # 6.2.4 we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walker_list)):
            all_id.append(walker_list[i]["con"])
            all_id.append(walker_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # 6.2.5 initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        print('Spawned %d vehicles and %d walkers.' % (len(vehicle_list), len(walker_list)))
        
        # 7. Define spawn point (manual) for test agents
        transform = random.choice(world.get_map().get_spawn_points())
        transform_2 = carla.Transform(carla.Location(x=2.5, y=1.1, z=0.7))
        # 7.1 Spawn test agents to simulation
        test_agent = world.try_spawn_actor(test_agent_bp, transform)
        test_agent.set_autopilot(True)
        test_agent.apply_control(carla.VehicleControl(gear=1, throttle=1.0, steer=0.5, hand_brake=False))
#        test_agent.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, hand_brake=False))
        # 7.1.1 Setting Traffic Manager to test agent (04162020)
        tm.distance_to_leading_vehicle(test_agent, 1)
        tm.ignore_walkers_percentage(test_agent, 80)
        tm.ignore_vehicles_percentage(test_agent, 90)
        tm.ignore_lights_percentage(test_agent, 90)
#        tm.auto_lane_change(test_agent, True)
        tm.force_lane_change(test_agent, True)
#        tm.vehicle_percentage_speed_difference(test_agent, -20)
        actor_list.append(test_agent)
        # 7.2 Spawn sensor, attach to test vehicle
        test_cam = world.try_spawn_actor(test_cam_bp, transform_2, attach_to=test_agent)
#        test_cam.listen(lambda image: process_img(image))
        test_cam.listen(lambda image: predict_risk_img(image, output_dir))
        actor_list.append(test_cam)

        print('spawned %d test agents, press Ctrl+C to exit.' % len(actor_list))

        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)

    finally:
        print("\nDestroying %d actors" % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])
        print("\nDestroying %d vehicles" % len(vehicle_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicle_list])        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        
        print("\nDestroying %d walkers" % len(walker_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in all_id])        
        print("\n Reset traffic lights")
        tm.reset_traffic_lights()
        # Stop recording
        print("\nStop recording")
        client.stop_recorder()
#        generate_data()
        print("\nAll cleaned up.")

# Main method
def main():
    # 1. Set up CARLA client connection
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    try:
        parser = argparse.ArgumentParser(description='dataset_maker')
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2', help='directory where the dataset will be created')
        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2\training\Town01\Phase 7', help='directory where the dataset will be created')
        args = parser.parse_args()
        output_dir = args.output_dir
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        # 2.1
        # Print out available maps
#        print(client.get_available_maps())
#        print(random.choice(client.get_available_maps()).split("/")[4])
        
        # 3. Retrieve world from CARLA simulation
#        world = client.load_world("Town01")
        world = client.get_world()
#        world = client.load_world(random.choice(client.get_available_maps()).split("/")[4])
        print(world.get_map().name)
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        # 3.2 Create traffic manager client
        tm = client.get_trafficmanager() # default port = 8000
        # 3.3 Setting synchronous mode (04172020)
#        synchronous_master = False
#        settings = world.get_settings()
        tm.set_synchronous_mode(True)
        '''
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False 
        '''
        # 4. Create test agents
        test_agent_bp = blueprint_library.filter('vehicle.*')
        test_agent_bp = random.choice([x for x in test_agent_bp if int(x.get_attribute('number_of_wheels')) == 4])
        test_cam_bp = blueprint_library.find('sensor.camera.rgb')
        # 4.1 Print test agents blueprints
        print(test_agent_bp)
        print(test_cam_bp)
        
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
                        
        world.wait_for_tick()
        
        # 7. Define spawn point (manual) for test agents
        transform = random.choice(world.get_map().get_spawn_points())
        transform_2 = carla.Transform(carla.Location(x=2.5, y=1.1, z=0.7))
        # 7.1 Spawn test agents to simulation
        test_agent = world.try_spawn_actor(test_agent_bp, transform)
        test_agent.set_autopilot(True)
        test_agent.apply_control(carla.VehicleControl(gear=2, throttle=1.0, steer=0.0, hand_brake=False))
#        test_agent.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0, hand_brake=False))
        # 7.1.1 Setting Traffic Manager to test agent (04162020)
#        tm.distance_to_leading_vehicle(test_agent, 1)
        tm.ignore_walkers_percentage(test_agent, 70)
        tm.ignore_vehicles_percentage(test_agent, 70)
        tm.ignore_lights_percentage(test_agent, 60)
        tm.auto_lane_change(test_agent, False)
        tm.force_lane_change(test_agent, False)
#        tm.vehicle_percentage_speed_difference(test_agent, -20)
        actor_list.append(test_agent)
        # 7.2 Spawn sensor, attach to test vehicle
        test_cam = world.try_spawn_actor(test_cam_bp, transform_2, attach_to=test_agent)
#        test_cam.listen(lambda image: process_img(image))
        test_cam.listen(lambda image: predict_risk_img(image, output_dir))
        actor_list.append(test_cam) 
        
        print('spawned %d test agents' % len(actor_list))
        
        while True:
#            generate_data()
#            predict_traffic_risk()
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)
        
    finally:
#        tm.reset_traffic_lights()
        print("\nDestroying %d actors" % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])
        # Stop recording
        print("Stop recording")
        client.stop_recorder()
        generate_data()
        print("\nAll cleaned up.")

# 03202020
def generate_data():
        # 1.1 Initialize TRIP module (dataset generator)
        parser = argparse.ArgumentParser(description='dataset_maker')
        parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
        parser.add_argument('--object_model_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\accident_KitDashV_6000.npz')
        parser.add_argument('--object_label_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\obj.names') # must be specified other than 'coco' and 'voc'    
        parser.add_argument('--object_cfg_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\yolo-obj.cfg')
        parser.add_argument('--object_detection_threshold', type=float, default=0.1)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--save_img', type=bool, default=True, help='save_img option')
        parser.add_argument('--video', type=bool, default=False, help='video option')
#        parser.add_argument('--input_dir', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip', help='input directory') # 04162020
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip', help='directory where the dataset will be created')
#        parser.add_argument('--input_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2', help='input directory')
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2', help='directory where the dataset will be created')
        parser.add_argument('--input_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2\training\Town05', help='input directory')
        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2\training\Town05', help='directory where the dataset will be created')
#        parser.add_argument('--layer_name_list', default='conv33,conv39,conv45', help='list of hidden layers name to extract features')
        parser.add_argument('--layer_name_list', default='conv33', help='list of hidden layers name to extract features')
        args = parser.parse_args()
        save_img = args.save_img
        output_dir = args.output_dir
        input_dir = args.input_dir
        layer_name_list = args.layer_name_list.split(',')

        # Process spatio-temporal information for captured image
        yolov3_predictor = ObjectDetector(args.object_model_type, args.object_model_path, 
                                         args.object_label_path, args.object_cfg_path, args.object_detection_threshold,
                                     device=args.gpu)    
        # Dataset generation
        orig_input_dir = input_dir
        orig_output_dir = output_dir
        video_files = os.listdir(input_dir)
        for video_file in video_files:
            if video_file[-5:-2]>='0':                
                print(video_file)
                print('save %s feature...' % video_file)
                input_dir = orig_input_dir + '/' + video_file + '/orig_img'
                output_dir = orig_output_dir + '/' + video_file
                # フォルダが無ければ新規作成 / Create a new folder if there is none previously
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)
                for layer in layer_name_list:
                    if not os.path.isdir(os.path.join(output_dir, layer)):
                        os.mkdir(os.path.join(output_dir, layer))
                if save_img and not os.path.isdir(os.path.join(output_dir, 'img')):
                    os.mkdir(os.path.join(output_dir, 'img'))
                if not os.path.isdir(os.path.join(output_dir, 'ebox')):
                    os.mkdir(os.path.join(output_dir, 'ebox'))

                # 画像ファイルのリストを作成 / Create a list of image files
                print('load image...')
                file_list = os.listdir(input_dir)
                img_files = [f for f in file_list if os.path.isfile(os.path.join(input_dir, f))]                
                # 最初の画像を読み込み / Loading the first image
                orig_img = cv2.imread(os.path.join(input_dir, img_files[0]))
                # 基準となる画像の高さと幅を取得
                img_h, img_w = orig_img.shape[:2]
                # ファイル数分繰り返す / Repeat files for a few minutes 
                for img_file in img_files:
                     # 拡張子とファイル名を分ける / Separate file names from extensions
                    file, ext = os.path.splitext(img_file)
                    # 画像読み込み / load image
                    orig_img = cv2.imread(os.path.join(input_dir, img_file))
                    # サイズが異なる場合は変形してから入力 / If the size is different, deform and then enter
                    if (img_h, img_w) != orig_img.shape[:2] :
                        orig_img = cv2.resize(orig_img, (img_w, img_h))

                    bboxes, labels, scores, layer_ids, features = yolov3_predictor(orig_img)    
                    DatasetGenerator.save_images(orig_img, bboxes, output_dir, file) #10182019
                    DatasetGenerator.save_feature(features, layer_name_list, output_dir, file+'.npz')    
                    DatasetGenerator.save_ebox(bboxes, labels, layer_ids, img_h, img_w, output_dir, 'e'+file+'.txt')
            # End Dataset generation

# 03282020:
def predict_traffic_risk():
    # 1.1 Initialize TRIP module (dataset generator)
    parser = argparse.ArgumentParser(description='video_prediction')
#    parser.add_argument('--video_out_path', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip\test_carla_video.mp4')
    parser.add_argument('--video_out_path', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_traffic_manager_video.mp4')
#    parser.add_argument('--ds_path', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip')
    parser.add_argument('--ds_path', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_2')
    parser.add_argument('--ds_spec_file_name', default='ds_spec.txt')
    parser.add_argument('--layer_name',  choices=('conv33', 'conv39', 'conv45'), default='conv33') # must be specified other than 'coco' and 'voc'    
    parser.add_argument('--box_type', choices=('ebox', 'tbox'), default='ebox')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--model_param_file', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\model_ebox_param_carla.txt')
#    parser.add_argument('--plog_path', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\result\yolov3\dashcam\dashcam_carla_trip_elog.txt')
    parser.add_argument('--plog_path', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\dashcam_carla_traffic_manager_elog.txt')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()    
    video_out_path = args.video_out_path
    ds_path = args.ds_path
    ds_spec_file_name = args.ds_spec_file_name
    layer_name = args.layer_name
    box_type = args.box_type
    window_size = args.window_size
    model_param_file_path = args.model_param_file
    plog_path = args.plog_path
    gpu_id = args.gpu_id

    trip_predictor = TripVPredictor(ds_path, ds_spec_file_name, layer_name, box_type, window_size, model_param_file_path, plog_path, gpu_id)
    if video_out_path != '':
        trip_predictor.set_video_out(video_out_path) 
    trip_predictor.vpredict()   
 

if __name__ == '__main__':
    try:
        # Run main method
#        spawn_npc()
#        spawn_walker()
#        spawn_car()
#        spawn_motorbike()
#        spawn_bicycle()
#        main()
#        main_traffic_manager()
         main_traffic_manager_2()
#        generate_data()
#        predict_traffic_risk()
#        start_replay()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")


