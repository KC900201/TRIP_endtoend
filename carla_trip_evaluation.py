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
    sys.path.append(glob.glob(r'CARLA_0.9.7_project/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
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
from estimation.dataset_generator.dataset_generator_function import DatasetGenerator
from estimation.dataset_generator.object_detector import ObjectDetector

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960

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

def generate_dataset(image):
    parser = argparse.ArgumentParser(description='dataset_maker')
    parser.add_argument('--object_model_type', choices=('yolo_v2', 'yolo_v3'), default='yolo_v3')
    parser.add_argument('--object_model_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\accident_KitDashV_6000.npz')
    parser.add_argument('--object_label_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\obj.names') # must be specified other than 'coco' and 'voc'    
    parser.add_argument('--object_cfg_path', default=r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\estimation\model_v3\yolo-obj.cfg')
    parser.add_argument('--object_detection_threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_img', type=bool, default=True, help='save_img option')
    parser.add_argument('--video', type=bool, default=False, help='video option')
    parser.add_argument('--input_dir', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip', help='input directory')
    parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\TRIP_dataset\carla_trip', help='directory where the dataset will be created')
    parser.add_argument('--layer_name_list', default='conv33,conv39,conv45', help='list of hidden layers name to extract features')
    args = parser.parse_args()
    output_dir = args.output_dir
    
    # Process spatio-temporal information for captured image
    yolov3_predictor = ObjectDetector(args.object_model_type, args.object_model_path, 
                                     args.object_label_path, args.object_cfg_path, args.object_detection_threshold,
                                     device=args.gpu)    
    orig_img = image
    img_h, img_w = image.shape[:2]

    bboxes, labels, scores, layer_ids, features = yolov3_predictor(orig_img)
    DatasetGenerator.save_images(orig_img, bboxes, output_dir, image.frame) #10182019
    DatasetGenerator.save_feature(features, output_dir, image.frame+'.npz')    
    DatasetGenerator.save_ebox(bboxes, labels, layer_ids, img_h, img_w, output_dir, 'e'+image.frame+'.txt') #10182019

def predict_risk_img(image):
    # Process image
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    cv2.imshow("", i3)
    cv2.waitKey(2) # delay 2 seconds
    # save image
    generate_dataset(image)
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
    
# Spawn NPC function - 02292020
def spawn_npc():
    try:
        # 1. Set up CARLA client connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        
        # 2. Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        
        # 3. Retrieve world from CARLA simulation        
#        world = client.load_world(random.choice(client.get_available_maps()).split("/")[4])            
#        world = client.get_world()
        world = client.load_world('Town01')
        print(world.get_map().name)
#        world = client.get_world()
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
            vehicle.set_autopilot(True)
            vehicle_list.append(vehicle)            
        # ----------------------
        # 6.2 Spawn NPC walkers    
        # ----------------------
        # some settings
        percentagePedestriansRunning = 5.0      # how many pedestrians will run
        percentagePedestriansCrossing = 10.0     # how many pedestrians will walk through the road
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

# Main method
def main():
    try:
        # 1. Set up CARLA client connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
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
        # 3.1 Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        
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
#        test_agent.apply_control(carla.VehicleControl(gear=3, throttle=2.0, steer=0.0, hand_brake=True))
        test_agent.apply_control(carla.VehicleControl(throttle=1.0, steer=1.0, hand_brake=False))
        actor_list.append(test_agent)
        # 7.2 Spawn sensor, attach to test vehicle
        test_cam = world.try_spawn_actor(test_cam_bp, transform_2, attach_to=test_agent)
#        test_cam.listen(lambda image: process_img(image))
        test_cam.listen(lambda image: predict_risk_img(image))
        actor_list.append(test_cam) 
        
        print('spawned %d test agents' % len(actor_list))
        
        while True:
            world.wait_for_tick() # wait for world to get actors
            time.sleep(1)
        
    finally:
        print("\nDestroying %d actors" % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list])
        # Stop recording
        print("Stop recording")
        client.stop_recorder()
        print("\nAll cleaned up.")
        
if __name__ == '__main__':
    try:
        # Run main method
        spawn_npc()
#        main()
#        start_replay()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")


