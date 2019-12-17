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
2019/12/16 - New function to remove extra files in virtual data directory (file no. 51 - 75, copyfiles.py)
2019/12/17 - Modify new function to delete "img" folder in virtual data directory (copyfiles.py)