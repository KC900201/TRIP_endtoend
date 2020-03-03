TRIP research project
=======
Traffic Risk Provision Network project: end-to-end architecture
@Authors: atsumilaboratory
@Contributors: Atsumi Masuyasa, Murata Yuuki, Kwong Cheong Ng

Description
-----------

Requirements
------------
- Need to create a traffic scenario for evaluation
- Design some of simulated risky situation, safe situation in CARLA simulator for evaluation
- Implement TRIP risk prediction evaluation in CARLA simulator during training and testing
- Need to create a simulation dataset (based on images from CARLA). Output 
- 1st trial of end-to-end simulation (DL: 10/23/2019)
	- Use testimages (images_test) from 456001 -45610 - divided into 2 folders (50 img each)
        - run trial 1 time for each folder (from dataset generation - risk prediction, total 2 run for 2 folders)
        - Evaluate risk prediction for each folder

Task for IW-FCV2020
-------------------
Deadline: Next Wednesday (10/23)
1. Fine tune risk prediction
2. Run TRIP simulation on CARLA
3. 1st evaluation on risk prediction - risky and not risky

Updates
-------
2019/09/12 - Did the first commit and push to Github repository
2019/09/14 - Draft first src code for end-to-end architecture. Mind mapping next stage
2019/10/10 - Completed first draft of end-to-end architecture (estimation part)
2019/10/11 - Begun first trial of end-to-end architecture (estimation part) coding
2019/10/12 - Wrote a batch file to test run coding for end-to-end architecture. Commented risk prediction part.
2019/10/13 - Fixing the single backslash "\" problem during input file reading. Problem persists
2019/10/14 - Continue fixing input file reading problem. Attempted to replace double backslash
2019/10/16 - Fixed gpu and gpu_id attributes string conflict. Temporarily hardcode input and output directory for dataset generation
2019/10/17 - Fixed directory path. Test run dataset_generation part successfully. Need minor fix on data processing (unknown attr)
2019/10/21 - Tested successfully for dataset generation part. Able to create attributes for convolution 33, 39 and 45. 
	     Need to retrieve training attributes for risk prediction training.
2019/10/22 - Tested successfully the flow from dataset generation to risk prediction to video risk prediction. 
	     Encountered pathnotfound exception in trip_dataset.py (line 28) while running video prediction (trip_vpredictor -> trip_predictor -> trip_dataset) 
             due to coding style when reading spec_files directory and ds_spec). Need to change coding style to suit own file directory.
2019/10/23 - Amend directory file input for video prediction part (trip_combined.py - plog_path, ds_path; trip_predictor.py)
2019/10/24 - Finally test run 1st trial of end-to-end architecture. Omitted training (risk prediction) coding due to unnecessary code in the architecture. 
             Videos are able to be created. Future questions: Enhance the convolutional neural network layer coding in trip_lstm.py
2019/10/25 - Enhancement to include new LSTM (input_middle_conv) and new model arch (trip_c_lstm.py, trip_lstm.py) 
2019/10/26 - Enhance training part to run multiple param files, refine input middle convolution layer by reducing ksize, stride, pad and kernel
2019/10/27 - Increase one more convolutional layer (input_sec_middle_conv) in trip_c_lstm.py 
2019/10/28 - Increase one more lstm (lstm3) in trip_lstm.py 
2019/10/29 - Increase one conv layer (input_third_middle_conv) in trip_lstm.py without implementation. Edit param files to run LSTM3 architecture.
2019/11/01 - Revert batch size to 32 in training (endtoend_spec.txt). 
	     Create new param files (model_tbox_lstm3_param1 - ...param5.txt) for training without drop-out mode.
	     Remove max_pooling at extra input middle conv layer to prevent data loss, insert dropout in between middle conv layer
2019/11/18 - Create new files for csv to txt conversion and file copying. Renamed training config
2019/11/19 - New file to convert GTAV dataset annotation files (csv) to yolo bbox annotation
2019/11/20 - Modifications for gta2yolo.py
2019/11/29 - Modifications for copyfiles.py. Update new cfg file for yolo-obj.cfg using updated anchors
2019/11/30 - Enhance risk prediction training to have one more parameter for virtual data input
2019/12/04 - Remove virtual data set test directory (trip_trainer.py). Add new function to move and separate files from train0 folder (copyfiles.py)
             New python file to store folders name for training
2019/12/05 - Add new functions (learn_virtual, learn_mix) to train risk prediction with virtual data (trip_trainer.py). 
	     Comment dataset generation part and append new functions (trip_combined.py)
2019/12/08 - Add new function to count number of files (copyfiles.py)
2019/12/10 - Comment out input for video prediction path, include one input for choosing training data (trip_combined.py; trip_trainer.py)
2019/12/11 - Enhance risk prediction training to have one more parameter for mix data input (real + virtual) - trip_combined.py; trip_trainer.py
2019/12/14 - New function to train virtual and real data separately (trip_trainer.py, trip_combined.py) 
2019/12/16 - New function to remove extra files in virtual data directory (file no. 51 - 75, copyfiles.py)
2019/12/17 - Modify new function to delete "img" folder in virtual data directory (copyfiles.py)
2019/12/18 - new model architecture function to test increasing accuracy (trip_c_lstm.py), modify RP training to evaluate accuracy increase at 25th epoch (trip_dataset.py)
2019/12/23 - remodify function to reduce accuracy value (trip_trainer.py)
2019/12/24 - New model architecture that reverts ReLu and Tanh activation (trip_c_lstm.py)
2019/12/26 - New function to separate A3D datasets (copyfiles.py)
2020/01/09 - Add in one features to extract only wanted feature files (skip interval) (trip_combined.py, trip_dataset.py, trip_trainer.py)
	   - Import cache to speed up process (trip_dataset.py)
2020/01/15 - Create plot graph table for vpredict result (make_result_table_vpredict.py);
	     Include graph for average risk estimation value
2020/01/22 - Remove pretraining codes (directory: training/*)
2020/01/28 - Add in check files for checking tensorflow (tensorflow_self_check.py). Add in new spec file (vpredict_spec.txt)
2020/01/31 - Include new folder and files for CARLA simulation (carla_sim/*)
2020/02/17 - Include new library files for CARLA (carla-0.9.7-py3.7-win-amd64.egg),
  	     New files for trial testing carla (carla_sim/PythonAPI/KC_test/test_1.py)
             New test file for trial testing carla (trial_1.py)
2020/02/18 - New test file for CARLA coding familiarization (KC_TEST\test_2.py)
2020/02/19 - Generate NPC and alter dynamic weather (test_2.py)
2020/02/20 - Spawn pedestrian NPC and control NPC dynamics 
2020/02/23 - New test file for testing pedestrian (test_3.py)
2020/02/24 - Modify spawn walker code
2020/02/25 - Edit src code to fix camera angle, spawn actors. Further improve walker npc (test_2.py) 
2020/03/02 - Predict maximum risk function in between n frames (trip_lstm.py)


