# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:23:23 2020

@author: KwongCheongNg, atsumilab
@filename: carla_trip_evaluation_2.py
@coding: utf-8
@description: Improved evaluation code from previous for testing TRIP risk prediction module on CARLA-based simulated environment. 4 main scenarios are to be tested:
              1. Near miss 
              2. Traffic accident
              3. Pedestrian / vehicle crossing at road intersection
              4. Non-accident (safe) incident
              Based code refers to previous src code and manual_control.py
              Recording will done during experiment to record evaluation progress and will be further used to train proposed module
@Reference: manual_control.py; carla_trip_evaluation.py
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/,
      https://pythonprogramming.net/control-camera-sensor-self-driving-autonomous-cars-carla-python/?completed=/introduction-self-driving-autonomous-cars-carla-python/
      RGB color setting: https://www.w3schools.com/colors/colors_picker.asp?colorhex=9c7ec9
========================
Date          Comment
========================
05252020      First revision 
05252020      Include Traffic Manager module for traffic scenario manipulation
05262020      Create NPC spawning - car; bikes and bicycles; pedestrians
              Reset Traffic Manager module
06122020      Revert NPC spawning method to not classify type of vehicles 
              Default set sef.player to be as 4 wheel vehicles
06162020      Split NPC spawning and self.player spawning        
"""

from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# -- Switch to factory version of CARLA v0.9.8 ---------------------------------
# ==============================================================================

import glob
import os
import sys

try:
    sys.path.append(glob.glob(r'CARLA_0.9.8/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % ( # 05252020
#    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- import TRIP modules -------------------------------------------------------
# ==============================================================================
from estimation.dataset_generator.dataset_generator_function import DatasetGenerator # 03132020
from estimation.dataset_generator.object_detector import ObjectDetector # 03132020
from risk_prediction.trip_vpredictor import TripVPredictor

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import collections
import cv2
import datetime
import logging
import math
import random
import re
import time
import weakref

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
COLLISION_INFO = ''
TOWN_MAP = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07']

# Create list of actors for test agent, NPC (vehicles, pedestrians)
#actor_list = []
#vehicle_list = []
#walker_list = []
#all_id = []
#car_list = []
#bike_list = []

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

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

def predict_risk_img(image, output_dir):
    # Process image
    i = np.array(image.raw_data)
    i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    i3 = i2[:, :, :3] # height, width, first 3 rgb value
    
    # 05212020
    font = cv2.FONT_HERSHEY_COMPLEX
    bottomleft = (10,500)
    fontScale = 10
    fontColor = (0,0,0)
    lineType = 2
        
    # 05232020
    print(len(COLLISION_INFO))
    img = cv2.putText(i3,str(COLLISION_INFO), 
        bottomleft, 
        font, 
        fontScale,
        fontColor,
        lineType)  
    # end 05212020
#    cv2.imwrite("out.jpg", i3)
    cv2.imshow("image", img)
    
    cv2.waitKey(1) # delay 1 seconds
    # save image
#    image.save_to_disk(output_dir + '/test/orig_img/%08d' % image.frame)
#    image.save_to_disk(output_dir + '/Town01/orig_img/%08d' % image.frame)
    image.save_to_disk(output_dir + '/orig_img/%08d' % image.frame)
    return i3 / 255.0

# Retrieve collision info
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_collision_info(event):
    actor_type = get_actor_display_name(event.other_actor)
    COLLISION_INFO = 'Collision with ' + str(actor_type)
#    print('Collision with %r' % actor_type)
    print(COLLISION_INFO)
    
# Function to return percentage (int) of whole number
# Used to delegate ratio of NPC spawning by category (car, bike, pedestrian)
def percentage(percent, whole):
    return int((percent * whole) / 100.0)

# Function to generate prediction video (spatio-temporal) from TRIP
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

# Function to generate spatio-temporal info from captured images
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

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_client, carla_world, traffic_manager, hud, args):
        self.client = carla_client
        self.world = carla_world
        self.actor_role_name = args.rolename
        self.traffic_manager = traffic_manager # 05252020
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        # 05262020
        self.npc_car = []
        self.npc_bike = []
        self.npc_walker = []
        self.npc_walker_id = []
        self.all_actors = None
        # End 05262020
        self.reset_traffic_manager()
        self.spawn_npc()
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
#        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        # 06012020
        blueprint = self.world.get_blueprint_library().filter(self._actor_filter)
        blueprint = [x for x in blueprint if int(x.get_attribute('number_of_wheels')) == 4]
        blueprint = [x for x in blueprint if not x.id.endswith('isetta')]
        blueprint = [x for x in blueprint if not x.id.endswith('carlacola')]
        blueprint = random.choice(blueprint)
        # end 06012020
        blueprint.set_attribute('role_name', self.actor_role_name)        
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # 05252020
            self.traffic_manager.ignore_walkers_percentage(self.player, 80)
            self.traffic_manager.ignore_vehicles_percentage(self.player, 70)
            self.traffic_manager.ignore_lights_percentage(self.player, 70)
            self.traffic_manager.auto_lane_change(self.player, True)        # Set up the sensors.
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # 05252020
            self.traffic_manager.ignore_walkers_percentage(self.player, 70)
            self.traffic_manager.ignore_vehicles_percentage(self.player, 70)
            self.traffic_manager.ignore_lights_percentage(self.player, 65)
#            self.traffic_manager.vehicle_percentage_speed_difference(self.player, -10) # 04062020
#            self.traffic_manager.distance_to_leading_vehicle(self.player, 10)
            self.traffic_manager.auto_lane_change(self.player, True)        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        '''
        #---------------------
        # Spawn NPC car
        #---------------------
        for i in range(npc_car_amt):
            transform = random.choice(self.world.get_map().get_spawn_points())
            blueprint_car = random.choice([x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4])
            # Set attribute for car blueprint
            if blueprint_car.has_attribute('color'):
                color = random.choice(blueprint_car.get_attribute('color').recommended_values)
                blueprint_car.set_attribute('color', color)
            if blueprint_car.has_attribute('driver_id'):
                driver_id = random.choice(blueprint_car.get_attribute('driver_id').recommended_values)
                blueprint_car.set_attribute('driver_id', driver_id)
            blueprint_car.set_attribute('role_name', 'autopilot')
            
            car = self.world.try_spawn_actor(blueprint_car, transform)
            if not (isinstance(car, type(None))): # 05132020
                car.set_autopilot(enabled=True)
                self.traffic_manager.ignore_lights_percentage(car, 90) # 04062020
                self.traffic_manager.vehicle_percentage_speed_difference(car, -20) # 04062020
                self.traffic_manager.distance_to_leading_vehicle(car, 30)
                self.traffic_manager.ignore_walkers_percentage(car, 80)
                self.traffic_manager.ignore_vehicles_percentage(car, 90)
                self.traffic_manager.auto_lane_change(car, False)                self.npc_car.append(car)
        #-----------------------------
        # Spawn NPC bikes and bicycles
        #-----------------------------
        for i in range(0, npc_bike_amt):
            transform = random.choice(self.world.get_map().get_spawn_points())
            blueprint_bike = random.choice([x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 2])
            # Set attribute for bike blueprint
            if blueprint_bike.has_attribute('color'):
                color = random.choice(blueprint_bike.get_attribute('color').recommended_values)
                blueprint_bike.set_attribute('color', color)
            if blueprint_bike.has_attribute('driver_id'):
                driver_id = random.choice(blueprint_bike.get_attribute('driver_id').recommended_values)
                blueprint_bike.set_attribute('driver_id', driver_id)
            blueprint_bike.set_attribute('role_name', 'autopilot')

            bike = self.world.try_spawn_actor(blueprint_bike, transform)
            if not (isinstance(bike, type(None))): # 05132020
                bike.set_autopilot(enabled=True)
#                self.traffic_manager.ignore_lights_percentage(bike, 90) # 04062020
#                self.traffic_manager.vehicle_percentage_speed_difference(bike, -20) # 04062020
#                self.traffic_manager.distance_to_leading_vehicle(bike, 30)
#                self.traffic_manager.ignore_walkers_percentage(bike, 80)
#                self.traffic_manager.ignore_vehicles_percentage(bike, 90)
#                self.traffic_manager.auto_lane_change(bike, False)
                self.npc_bike.append(bike)
        '''
    # 06062020        
    def spawn_npc(self):
        # Create NPC blueprint
        npc_vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')
        npc_walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        # Avoid spawning NPC prone to accident
#        npc_vehicle_bp = [x for x in npc_vehicle_bp if int(x.get_attribute('number_of_wheels')) == 4]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('isetta')]
        npc_vehicle_bp = [x for x in npc_vehicle_bp if not x.id.endswith('carlacola')]
        # ---------------------
        # Spawn NPC vehicle    
        # ---------------------
        spawn_points = self.world.get_map().get_spawn_points()
        num_spawn_points = len(spawn_points)
        print("Number of spawn points: %d" % int(num_spawn_points))
        npc_amt = percentage(40, num_spawn_points) # 05152020
#        npc_car_amt = percentage(20, num_spawn_points)
#        npc_bike_amt = percentage(60, num_spawn_points)
        if npc_amt <= num_spawn_points:
            random.shuffle(spawn_points)
        elif npc_amt > num_spawn_points:
            msg = 'Requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, npc_amt, num_spawn_points)
            print('Requested %d vehicles, but could only find %d spawn points' % (npc_amt, num_spawn_points))
            npc_amt = int(num_spawn_points / 2)  # Assign half number of spawn points to NPC to prevent spawning error
        # 06012020
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
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if not (isinstance(vehicle, type(None))): # 05132020
                vehicle.set_autopilot(enabled=True)
                self.traffic_manager.ignore_lights_percentage(vehicle, 70) # 04062020
#                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 10) # 04062020
#                self.traffic_manager.distance_to_leading_vehicle(vehicle, -10)
                self.traffic_manager.ignore_walkers_percentage(vehicle, 90)
                self.traffic_manager.ignore_vehicles_percentage(vehicle, 80)
                self.traffic_manager.auto_lane_change(vehicle, True)
                self.npc_car.append(vehicle)            
        # End 06012020
        #----------------------
        # Spawn NPC walkers    
        #----------------------      
        # Some settings
        percentagePedestriansRunning = 35.0      # how many pedestrians will run
        percentagePedestriansCrossing = 70.0     # how many pedestrians will walk through the road
        # Take all random locations to spawn
        spawn_points = []
        npc_walker_amt = percentage(55, num_spawn_points)
        
        for i in range(npc_walker_amt):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # Spawn walker object
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
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.npc_walker.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 6.2.3 spawn walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.npc_walker)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), self.npc_walker[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.npc_walker[i]["con"] = results[i].actor_id
        # 6.2.4 we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.npc_walker)):
            self.npc_walker_id.append(self.npc_walker[i]["con"]) 
            self.npc_walker_id.append(self.npc_walker[i]["id"])
        self.all_actors = self.world.get_actors(self.npc_walker_id)

        # 6.2.5 initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.npc_walker_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
            
#        print('Spawned %d cars, %d bikes and %d walkers.' % (len(self.npc_car), len(self.npc_bike), len(self.npc_walker)))
        print('Spawned %d NPC vehicles and %d walkers.' % (len(self.npc_car), len(self.npc_walker)))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])
    
    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None
    
    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None
    # 05262020
    def reset_traffic_manager(self):
        print("\nReset traffic lights")
        self.traffic_manager.reset_traffic_lights()

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
#            self.gnss_sensor.sensor,
#            self.imu_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        # 05262020
        # Destroy NPC car
        print("\nDestroying %d NPC cars" % len(self.npc_car))
#        for car in self.npc_car:
#            if car is not None:
#                self.client.apply_batch_sync(carla.command.DestroyActor(car))
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.npc_car])
        # Destroy NPC bike
        '''
        print("\nDestroying %d NPC bikes" % len(self.npc_bike))
#        for bike in self.npc_bike:
#            if bike is not None:
#                self.client.apply_batch_sync(carla.command.DestroyActor(bike))
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.npc_bike])
        '''
        # Destroy NPC walkers
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.npc_walker_id), 2):
            self.all_actors[i].stop()
        print("\nDestroying %d NPC walkers" % len(self.npc_walker))
#        for walker_id in self.npc_walker_id:
#            if walker_id is not None:
#                self.client.apply_batch_sync(carla.command.DestroyActor(walker_id))
        self.client.apply_batch_sync([carla.command.DestroyActor(x) for x in self.npc_walker_id])
        
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
#            self._autopilot_enabled = False
            self._autopilot_enabled = True # Set autopilot at start
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= carla.VehicleLightState.All ^ carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= carla.VehicleLightState.All ^ carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.collision_info = None # 05262020
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        self.collision_info = 'Collision with ' + str(actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.collision_info = None
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        parser = argparse.ArgumentParser(description='dataset_maker')
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_3\training\Town03\Phase 2', help='directory where the dataset will be created')
#        parser.add_argument('--output_dir', default=r'C:\Users\atsumilab\Pictures\CARLA_dataset\test_3\training\Town07\T7_S3', help='directory where the dataset will be created')
        parser.add_argument('--output_dir', default=r'E:\TRIP\Datasets\CARLA_dataset\test_3\original_files\Town01\T1_S13', help='directory where the dataset will be created')
 
        args = parser.parse_args()
        output_dir = args.output_dir

        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
#            image.save_to_disk('_out/%08d' % image.frame)
            # Process image
            '''
            i = np.array(image.raw_data)
            i2 = i.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            i3 = i2[:, :, :3] # height, width, first 3 rgb value

            font = cv2.FONT_HERSHEY_COMPLEX
            bottomleft = (10,500)
            fontScale = 10
            fontColor = (0,0,0)
            lineType = 2
            img = cv2.putText(i3,str(self.collision_info), 
                    bottomleft, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType) 
            cv2.imwrite("out.jpg", img)
            '''
            image.save_to_disk(output_dir + '/orig_img/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    traffic_manager = None
    
    client = carla.Client(args.host, args.port)
    client.set_timeout(3.0)
    traffic_manager = client.get_trafficmanager()
    hud = HUD(args.width, args.height)
   
#    world = World(client, client.load_world(TOWN_MAP[0]), traffic_manager, hud, args)
    world = World(client, client.get_world(), traffic_manager, hud, args)
    
    try:
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

#        hud = HUD(args.width, args.height)
#        world = World(client.get_world(), hud, args)
#        world = World(client, client.load_world(TOWN_MAP[2]), traffic_manager, hud, args)

        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
            # Reset traffic manager module in simulation
            world.reset_traffic_manager() 
        pygame.quit()

# ==============================================================================
# -- Main Function ---------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        default=True,
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print("\ndone.")