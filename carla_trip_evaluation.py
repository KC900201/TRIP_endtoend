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
"""
import sys
