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