Author: Ng Kwong Cheong (ケーシー), Murata （村田）
Date: 24/01/2019

Details
-----------
1. An attempt to update existing dataset_generator (2.1) from YOLOv2 to YOLOv3 model
2. Wish to modify src code to resume TRIP experiment with wholely YOLOv3 object detection model
3. Target completion date: Before 8/2/2019 to catch up with JSAI Feb 2019 submission date

Source code
--------------
Folder: HDD ->HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\dataset_generator3.0\dataset_generator
1. dataset_generator.py - generate feature extractions and save each annotation of video images. Main src file
                        - functions: save_images()
                                     save_feature()
                                     save_ebox()
                                     **save_specfile() - not used
                                     copy_file()
2. yolo_base_feature_ext.py > yolo base code to retrieve bounding box, objects, and configs. take note "cuda" library to use cpu
3. yolov3_feature_ext.py > src code using yolo_base_feature_ext.py to perform feature extraction. Determines which layer to extract which size object
4. yolov2.py -> used by former dataset_generator (2.1, 2.0) in extracting features based on yolov2 network
5. yolov2_darknet.py -> used former dataset_generator (2.1, 2.0) in extracting features based on yolov2 network, 21st layer is used

Result
--------
Result folder: HDD ->HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\dataset_generator3.0\dataset_generator\test
1. Folders 00001 to 00045
2. Subfolders conv33, conv39, conv4 -> contains .npz files for risk prediction (33: big objects, 39: medium sized, 45: small size)
3. subfoler ebox -> annotation files of estimation box-based.

Risk Prediction Training
-------------------------
Directory: HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\TrafficRiskPrediction1.2-180106 
Spec file path: HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam
Training images path (YOLOv3): E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3 -> 4 folders: test0, test1, train0, train1
Spec files: 1) model_eboz_param.txt
	    2) training_spec.txt
	    3) dashcam_ebox_elog.txt (HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\result)
model file: accident_KitDash_8000.txt (HD-GDU3:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\model)
