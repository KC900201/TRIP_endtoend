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
02192020      Spawn npc (vehicles), alter weather conditions
02202020      Spawn npc (pedestrian, spectator), edit NPC attributes
02252020      Rectify pedestrian NPC attributes
"""

import sys
import os
import glob
import random
import time
import cv2
import numpy as np
import logging

try:
    #sys.path.append(glob.glob(r'C:\Users\atsumilab\Documents\Projects\TRIP_endtoend\carla_sim\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob(r'../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
NPC_AMT = random.randint(1, 400) # amount of npc (02192020)

# Create actor list
actor_list = []
all_id = []

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

# Function to spawn NPC (02192020)
def spawn_npc(world):
    # 02192020
    blueprint_library = world.get_blueprint_library()
    npc_bp = blueprint_library.filter('vehicle.*')
    
    # Spawn points for NPC (02192020)
    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    number_of_npc = NPC_AMT
    
    print("Number of spawn points: %d" % int(number_of_spawn_points))
    print("Number of NPC: %d" % int(number_of_npc))
    
    if number_of_npc <= number_of_spawn_points:        
        random.shuffle(spawn_points)
    else:
        number_of_npc = number_of_spawn_points

    # Spawn NPC (02192020)
    # @todo cannot import these directly
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_npc:
            break
        blueprint = random.choice(npc_bp)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

    return batch

# Function to spawn pedestrian(world) - 02/20/2020
def spawn_walker(world):
    
    blueprint_library = world.get_blueprint_library()
    walker_bp = blueprint_library.filter('walker.pedestrian.*')
    
    # walker speed setting
    percentagePedestriansRunning = 0.0 # how many pedestrians will run
    percentagePedestriansCrossing = 0.0 # how many pedestrians will cross roads    
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
    walker_speed = []
    for n, transform in enumerate(spawn_points):
        if n >= number_of_npc:
            break
        blueprint = random.choice(walker_bp)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        if blueprint.has_attribute('speed'):
            if(random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(blueprint.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(carla.command.SpawnActor(blueprint, transform))
        
    return batch

# Function to change weather (02192020)
def dynamic_weather(world):
    # Set weather parameters (02192020)
    print(world.get_weather())
    
    new_weather = carla.WeatherParameters(
            cloudyness=80.45,
            precipitation=33.33,
            precipitation_deposits=20.0,
            wind_intensity = 50,
            sun_azimuth_angle=90)
    
    world.set_weather(new_weather)
    
    print(world.get_weather())

# Test create walker - 02/20/2020
def main_walker():
    try:
        # Set up connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
        
        # Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
                
        # Retrieve world map
        world = client.get_world() # retrieve map
        blueprint_library = world.get_blueprint_library()
        
        # Blueprint for walker (pedestrian)
        walker_bp = blueprint_library.filter('walker.pedestrian.*')

        # Create list for walkers, walkers speed
        walkers_list = []
        batch = []
        walker_speed = []
        walker_speed2 = []
        
        # walker speed setting
        percentagePedestriansRunning = 0.0 # how many pedestrians will run
        percentagePedestriansCrossing = 0.0 # how many pedestrians will cross roads    
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

        for n, spawn_point in enumerate(spawn_points):
            if n >= number_of_npc:
                break
            blueprint = random.choice(walker_bp)
            # set as not invencible
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            # set the max speed
            if blueprint.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(blueprint.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(blueprint.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(blueprint, spawn_point))
        
        results = client.apply_batch_sync(batch, True)
        
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2 
        
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)
        
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d walkers, press Ctrl+C to exit.' % len(walkers_list))

        while True:
            world.wait_for_tick()
        
    finally:
        print("\n Destroying %d walkers" % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        print("All cleaned up!")

# Test create actor spectator - 02/20/2020
def main_spectator():
    try:
        # Set up connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
        
        # Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
                
        # Retrieve world map
        world = client.load_world('Town03') # load map town 01

        # Create spectator - 02/20/2020
        spectator = world.get_spectator()

        # Retrieve blueprint
        blueprint_library = world.get_blueprint_library()

        bmw_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))
        
        bmw_bp.set_attribute('color', random.choice(bmw_bp.get_attribute('color').recommended_values))
        bmw_bp.set_attribute('role_name', 'autopilot')
        
        # Define spawn point (manual) for actor
        transform = random.choice(world.get_map().get_spawn_points())
        
        # Spawn actor
        bmw = world.try_spawn_actor(bmw_bp, transform)
        bmw.set_autopilot(True)
        bmw.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        actor_list.append(bmw)

        # Wait for world to get actors
        #world.tick()

        while True:
            #world_snapshot = world.wait_for_tick()
            world.wait_for_tick()
            #print("BMW ID: %d " % int(bmw.id))
            #actor_snapshot = world_snapshot.has_actor(bmw.id)
        
        # Set spectator at given transform (vehicle transform) - 02/20/2020
        spectator.set_transform(bmw.get_transform())

        #time.sleep(240)
        
    finally:
        print("\n Destroying %d actors" % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("All cleaned up!")

def main():
    try:
        # Set up connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(3.0)
        
        # Start logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
                
        # Retrieve world map
        #world = client.load_world('Town04') # load map town 01
        world = client.get_world()
        #dynamic_weather(world)
                        
        # Retrieve blueprint
        blueprint_library = world.get_blueprint_library()
        
        # Create actors
        audi_bp = random.choice(blueprint_library.filter('vehicle.audi.*'))
        #audi_bp = blueprint_library.find('vehicle.tesla.model3')
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Load blueprint attributes
        print(audi_bp)
        print(camera_bp)
        
        # Set attributes for actors
        audi_bp.set_attribute("color", "0, 255, 0")
        audi_bp.set_attribute("role_name", "autpilot")
#        audi_bp.set_attribute("sticky_control", "false")
        
        camera_bp.set_attribute("image_size_x", f"{IMAGE_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IMAGE_HEIGHT}")
        camera_bp.set_attribute("fov", f"110")
#        camera_bp.set_attribute("iso", f"1200")
        
        # Define spawn point (manual) for actor
        transform = random.choice(world.get_map().get_spawn_points())
        transform_2 = carla.Transform(carla.Location(x=2.5, y=1.1, z=0.7))

        # Spawn actor
        audi = world.try_spawn_actor(audi_bp, transform)
#        audi = carla.command.SpawnActor(audi_bp, transform).then(carla.command.SetAutopilot(carla.command.FutureActor, True))
        #audi = vehicle_control(audi) # change vehicle physics control
        audi.set_autopilot(True)
        audi.apply_control(carla.VehicleControl(throttle=2.0, steer=0.0))
        actor_list.append(audi)
        # Spawn sensor, attach to vehicle
        camera = world.try_spawn_actor(camera_bp, transform_2, attach_to=audi)
        camera.listen(lambda image: process_img(image))
        actor_list.append(camera)
        
        npc_batch = spawn_npc(world) # return NPC actors 
         
        for response in client.apply_batch_sync(npc_batch):
            if response.error:
                logging.error(response.error)
            else:
                actor_list.append(response.actor_id)
         
        print('spawned %d NPC actors' % len(actor_list))
                        
        while True:
            world.wait_for_tick() # wait for world to get actors
        
        # Delay time
        #time.sleep(600) # 4 minutes
        
    finally:
#        for actor in actor_list:
#            actor.destroy()
        print("\n Destroying %d actors" % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("All cleaned up!")

# Main function
if __name__ == '__main__':
    try:
#        main()
        main_walker()
#        main_spectator()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')