# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 2020

@author: atsumilab
@filename: test_1.py
@coding: utf-8
@description: First attempt to develop carla simulation from carla tutorial
@URL: https://carla.readthedocs.io/en/latest/python_api_tutorial/
========================
Date          Comment
========================
02172020      First revision 
"""

import glob
import os
import random
import sys

# Import carla library from path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Connect to Carla "client" object
client = carla.Client('localhost', 2000)
# Set time out - time limit to all networking op to prevent blocking
client.set_timeout(5.0) # seconds

# Retrieve world
world = client.get_world()

# Creating blueprints
blueprint_library = world.get_blueprint_library()

# Find specific blueprint.
collision_sensor_bp = blueprint_library.find('sensor.other.collision')
# Choose a vehicle blueprint at random
vehicle_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))
camera_bp = random.choice(blueprint_library.filter('sensor.camera.*'))

# Change the attributes of a blueprint (exp: car)
vehicle_bp.set_attribute('color', '255, 0, 0') # set as RGB
vehicle_bp.set_attribute('role_name', 'test_vehicle')
# Change attributes of a set of blueprints
vehicles = blueprint_library.filter('vehicle.*')
bikes = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 2]
for bike in bikes:
    bike.set_attribute('color', '0, 0, 255')

# Spawning actors
# We can find a spawn location from list of recommended transforms
transform = Transform(Location(x=230, y=195, z=40), Rotation(yaw=180))
#actor = world.spawn_actor(blueprint_library, transform)

# Retrieve spawn points
spawn_points = world.get_map().get_spawn_points()
# Spawn functions - argument to control whether actor is attached to another actor
camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle_bp) # camera attached to vehicle
camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))
# Handling actors
# We can move the actor and check its dynamic properties
location = vehicle_bp.get_location() # retrieve location
location.z += 10.0
vehicle_bp.set_location(location)
print(vehicle_bp.get_acceleration())
print(vehicle_bp.get_velocity())

# Freeze the actor
vehicle_bp.set_simulate_physics(False)
# Destroy the actor
vehicle_bp.destroy()

# Extra options
# Changing the weather
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=30.0,
    sun_altitude_angle=70.0
    )

world.set_weather(weather)
print(world.get_weather())

# World Snapshot
# Function to capture still images of carla world with a timestamp, record the location of every actor 

# Retrieve snapshot of world at this point in time
world_snapshot = world.get_snapshot()
# Wait for the next tick and retrieve snapshot of the tick
world_snapshot = world.wait_for_tick()
# Register a callback to get called every time we receive a new snapsho
# world.on_tick(lambda world_snapshot: do_something(world_snapshot))

## Map and Waypoints
# Retrieve map of current world
map = world.get_map()
print("Retrieve spawn points " + map.get_spawn_points())

# Waypoint - retrieve a waypoint on road closest to our vehicle
waypoint = map.get_waypoint(vehicle_bp.get_location())
# Nearest waypoint on center of a Driving or Sidewalk lane
waypoint_closest = map.get_waypoint(vehicle_bp.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
# Get current lane type
lane_type = waypoint_closest.lane_type
print(lane_type)
# Get available lane change
lane_change = waypoint_closest.lane_change
print(lane_change)

# Waypoints have func() to query next waypoints to return a list of waypoints at a certain distance 
# that can be accessed from current waypoint following traffic rules. Example below:

# Retrieve closest waypoint
waypoint = map.get_waypoint(vehicle_bp.get_locaation())

# Disable physics, in this example we're just teleporting the vehicle
vehicle.set_simulate_physics(False)

while True:
    # Find next waypoint 2 meters ahead
    waypoint = random.choice(waypoint.next(2.0)) # in meters
    print(waypoint)
    # Teleport vehicle
    vehicle.set_transform(waypoint.transform)

# Generate bulk waypoints all over the map
waypoint_list = map.generate_waypoints(2.0)
# Retrieve a topology graph of roads (tuple)

# Recording and Replaying System
# Start recording
client.start_recorder("recording01.log")
# Stop recording
client.stop_recorder()
# Replay
client.replay_file("recording01.log")