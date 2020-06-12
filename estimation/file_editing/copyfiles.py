"""
Created on Sat Sep 14 13:05:58 2019

@author: atsumilab
@filename: copyfiles.py
@code: utf-8
========================
Date          Comment
========================
11182019      First revision
11292019      Modifications to use glob library
12032019      New function to move files
12042019      Function to move file for virtual dataset allocation
12062019      New function to count no. of files in directory, for checking whether data transferred successfully
12162019      New function to remove extra files in virtual data directory (file no. 51 - 75)
12172019      Modify new function to delete "img" folder in virtual data directory
12262019      New function to separate A3D datasets (< 100 img and >= 100 img)
01082020      Temporary modifications to edit A3D datasets to separate accident and non-accident 
06072020      Function to rename all images to follow usual format for CARLA-based dataset
"""

import shutil
import errno 
import os
import stat
import glob
import math

from folder import Folder

folder_name = ['conv33', 'conv39', 'conv45', 'ebox', 'img', 'orig_img']
folder_name_2 = ['conv33', 'conv39', 'conv45', 'ebox', 'orig_img']
folder_name_a3d = ['conv33', 'conv39', 'conv45', 'ebox', 'img', 'orig_img', 'images']

def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

def ignore_function(ignore):
    def _ignore_(path, names):
        ignored_names = []
        if ignore in names:
            ignored_names.append(ignore)
        return set(ignored_names)
    return _ignore_

def copy(src, dest):
    try:
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns('*.py', '*.sh'))
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

def copyTree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if symlinks and os.path.islink(s):
      if os.path.lexists(d):
        os.remove(d)
      os.symlink(os.readlink(s), d)
      try:
        st = os.lstat(s)
        mode = stat.S_IMODE(st.st_mode)
        os.lchmod(d, mode)
      except:
        pass # lchmod not available
    elif os.path.isdir(s):
      copyTree(s, d, symlinks, ignore)
    else:
      shutil.copy2(s, d)

def moveTree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    if item in folder_name: # copy directory
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
          if os.path.lexists(d):
            os.remove(d)
          os.symlink(os.readlink(s), d)
          try:
            st = os.lstat(s)
            mode = stat.S_IMODE(st.st_mode)
            os.lchmod(d, mode)
          except:
            pass # lchmod not available
        elif os.path.isdir(s):
          moveTree(s, d, symlinks, ignore)
        else:
          shutil.copy2(s, d)
    else:         # move specific files
        filename = str(os.path.splitext(os.path.basename(item))[0])
        if "e" not in filename and "_" not in filename: # found no special characters in file name
            fileno = int(filename.lstrip().rstrip())
            if fileno > 50:
                new_fileno = int(fileno) - 50
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno == 50) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item.replace(filename, new_filename))
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTree(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d)
            else:
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTree(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d) 
        else: # found special characters in file name
            if 'e' in filename:
                fileno = int(filename.strip('e').lstrip().rstrip())
            else:
                fileno = int(filename.split("_")[0])
            if fileno > 50:
                new_fileno = int(fileno) - 50
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno == 50) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item.replace(filename, new_filename))
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTree(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d)
            else:                
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTree(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d)

def moveTreeVirtual(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
    os.makedirs(dst)
    shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    if item in folder_name: # copy directory
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
          if os.path.lexists(d):
            os.remove(d)
          os.symlink(os.readlink(s), d)
          try:
            st = os.lstat(s)
            mode = stat.S_IMODE(st.st_mode)
            os.lchmod(d, mode)
          except:
            pass # lchmod not available
        elif os.path.isdir(s):
          moveTreeVirtual(s, d, symlinks, ignore)
        else:
          shutil.copy2(s, d)
    else:         # move specific files
        filename = str(os.path.splitext(os.path.basename(item))[0])
        if "e" not in filename and "_" not in filename: # found no special characters in file name
            fileno = int(filename)
            if fileno > 75:
                new_fileno = fileno - 75
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno > 24) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item.replace(filename, new_filename)) 
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTreeVirtual(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d)                           
        else: # found special characters in file name
            if 'e' in filename:
                fileno = int(filename.strip('e').lstrip().rstrip())
            else:
                fileno = int(filename.split("_")[0])
            if fileno > 75:
                new_fileno = int(fileno) - 75
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno > 24) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item.replace(filename, new_filename))
                if symlinks and os.path.islink(s):
                  if os.path.lexists(d):
                    os.remove(d)
                  os.symlink(os.readlink(s), d)
                  try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                  except:
                    pass # lchmod not available
                elif os.path.isdir(s):
                  moveTreeVirtual(s, d, symlinks, ignore)
                else:
                  shutil.move(s, d)

# 12262019
def moveTreeA3D(src, dst, symlinks = False, ignore = None):
  lst = os.listdir(src)
  if ignore:
    excl = ignore(src, lst)
    lst = [x for x in lst if x not in excl]
  for item in lst:
    if item == folder_name_a3d[6]: # copy directory
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
          if os.path.lexists(d):
            os.remove(d)
          os.symlink(os.readlink(s), d)
          try:
            st = os.lstat(s)
            mode = stat.S_IMODE(st.st_mode)
            os.lchmod(d, mode)
          except:
            pass # lchmod not available
        elif os.path.isdir(s):
          for _, dirnames, filenames in os.walk(s):
            if len(filenames) >= 100:
                shutil.move(src, dst)

# 12162019
def delTreeVirtual(src, symlinks = False, ignore = None):
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
      if item in folder_name: # copy directory
        s = os.path.join(src, item)
        if item == folder_name[4]: # 12172019
            shutil.rmtree(s)
        else:
            if os.path.isdir(s):            
                delTreeVirtual(s, symlinks, ignore)
            else:
                continue
      else:
        filename = str(os.path.splitext(os.path.basename(item))[0])
        if "e" not in filename and "_" not in filename: # found no special characters in file name
            fileno = int(filename)
            if fileno > 50 and fileno <= 75:
                s = os.path.join(src, item)
                os.remove(s)
        else: # found special characters in file name
            if 'e' in filename:
                fileno = int(filename.strip('e').lstrip().rstrip())
            else:
                fileno = int(filename.split("_")[0])
            if fileno > 50 and fileno <= 75:
                s = os.path.join(src, item)
                os.remove(s)

def delTreeA3D(src, symlinks = False, ignore = None):
    lst = os.listdir(src)
    count = 1
    unwanted_file_length = len(lst) - 100
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
      new_fileno = count 
      if item in folder_name_a3d: # copy directory
        s = os.path.join(src, item)
        if item == folder_name_a3d[6]: # 12172019
#            shutil.rmtree(s)
            os.rename(s, os.path.join(src, "orig_img"))
        else:
            if os.path.isdir(s):            
                delTreeA3D(s, symlinks, ignore)
            else:
                continue
      else:
        filename = str(os.path.splitext(os.path.basename(item))[0])
        if "e" not in filename and "_" not in filename: # found no special characters in file name
            fileno = int(filename)
            if fileno > 100:
#            if fileno <= unwanted_file_length:
             s = os.path.join(src, item)
#            new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or fileno >= 100) else filename.replace(str(fileno), str(new_fileno))
#            os.rename(s, os.path.join(src, item.replace(filename, new_filename)))
#            count = count + 1
             os.remove(s)
        else: # found special characters in file name
            if 'e' in filename:
                fileno = int(filename.strip('e').lstrip().rstrip())
            else:
                fileno = int(filename.split("_")[0])
            if fileno > 100:
#            if fileno <= unwanted_file_length:                 
             s = os.path.join(src, item)
#            new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or fileno >= 100) else filename.replace(str(fileno), str(new_fileno))
#            os.rename(s, os.path.join(src, item.replace(filename, new_filename)))
#            count = count + 1
             os.remove(s)

def copyTreeFormat(src, dst, list, oname):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            if file_name in list:
                for odir in dst:
                    if os.path.isdir(odir):
#                        if odir.endswith(oname):
                        if str(os.path.basename(odir)) == oname:
                            new_odir = odir + "\\" + file_name + "_0" # for dashcam data
#                            new_odir = odir + "\\" + file_name  # for virtual data
                            copyTree(dir, new_odir)

def moveTreeFormat(src, dst, list, oname):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            file_name_ori = file_name.split("_")[0]
            if file_name_ori in list:
                for odir in dst:
                    if os.path.isdir(odir):
#                        if odir.endswith(oname):
                        if str(os.path.basename(odir)) == oname:
                            new_odir = odir + "\\" + file_name.replace("_0", "_1")
                            moveTree(dir, new_odir)

# 12262019
def moveTreeFormatA3D(src, dst, oname):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))            
            for odir in dst:
                if os.path.isdir(odir):
                    if odir.endswith(oname):
                        new_odir = odir + "\\" + file_name
#                        moveTreeA3D(dir, new_odir)
                        moveTree(dir, new_odir)

                        
def moveTreeFormatVirtual(src, dst, list, oname):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            if file_name in list:
                for odir in dst:
                    if os.path.isdir(odir):
                        if odir.endswith(oname):
                            new_odir = odir + "\\" + file_name
                            moveTreeVirtual(dir, new_odir)

def delTreeFormatVirtual(src, list):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            if file_name in list:
                    delTreeVirtual(dir)

def delTreeFormatA3D(src):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            delTreeA3D(dir)

# 12062019
def countFilesFolders(dir, list):
    files = 0
    #folders = 0
    
    if len(list) == 0:
        for _, dirnames, filenames in os.walk(dir):
            if os.path.basename(_) in folder_name_2:                
                files += len(filenames)
            #folders += len(dirnames)
    else:
        for _, dirnames, filenames in os.walk(dir):
            selected = os.path.basename(_)
            if selected in list:
                files += len(filenames)
                #folders += len(dirnames)
                for _, dirnames2, filenames2 in os.walk(_):
                    if os.path.basename(_) in folder_name_2: 
                        files += len(filenames2)
                    #folders += len(dirnames2)
    print('' + str(files) + ' files in '  + dir)
    #print('' + str(files) + ' files, ' + str(folders) + ' folders found ' + dir)

# 06072020
def rename_carla(dir):
    files = 0
    
    for _, dirnames, filenames in os.walk(dir):
        if os.path.basename(_) in folder_name:
            if os.path.basename(_) == folder_name[4]:
                shutil.rmtree(_)
            else:
                for i in range (len(filenames)):      
                    item = filenames[i]           
                    filename = str(os.path.splitext(os.path.basename(filenames[i]))[0])
                    if "_" in filename:
                        filename = filename.split("_")[0]
                    new_filename = ""
                    if filename.startswith("e"):
                        new_filename = str("e" + ('%06d' % (i+1)))
                    else: 
                        new_filename = str('%06d' % (i+1))
                    s = os.path.join(_, item)
                    os.rename(s, os.path.join(_, item.replace(filename, new_filename)))

if __name__ == '__main__':
    testdir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\test0\*')
    testdir1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\test1\*')
    traindir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\train0\*')
    traindir1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\train1\*')
    no_accident_dir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2\No_Accident\*')
    accident_car_dir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2\Accident_Car\*')
    accident_asset_dir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2\Accident_Asset\*')
    accident_pedestrian_dir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2\Accident_Pedestrian\*')
    accident_b_dir =  glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2\Accident_B\*')
    output_dir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\*')
    traindir_dashcam = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2019Q2\Dashcam_dataset\training\positive\*')
    testdir_dashcam = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2019Q2\Dashcam_dataset\testing\positive\*')
    traindir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\train0\*')
    traindir_1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\train1\*')
    testdir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\test0\*')
    testdir_1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\test1\*')
    vtraindir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\vtrain0\*')
    traindir_dashcam0 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\train0'
    traindir_dashcam1 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\train1'
#    traindir_dashcam0 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\train0'
#    traindir_dashcam1 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\train1'
    testdir_dashcam0 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\test0'
    testdir_dashcam1 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\test1'
#    testdir_dashcam0 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\test0'
#    testdir_dashcam1 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\test1'
    traindir_viena0 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\vtrain0'
    traindir_viena1 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\vtrain1'
#    traindir_viena0 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\vtrain0'
#    traindir_viena1 = r'D:\TRIP\Datasets\YOLO_KitDashV\ds4\vtrain1'
    viena_dir = r'E:\AtsumiLabMDS-2\TRIP\Dataset\VIENA2\image\Scenario2'
    dashcam_train = r'E:\AtsumiLabMDS-2\TRIP\Trip2019Q2\Dashcam_dataset\training\positive'
    dashcam_test = r'E:\AtsumiLabMDS-2\TRIP\Trip2019Q2\Dashcam_dataset\testing\positive'
#    traindir_mixed0 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain0'
#    traindir_mixed1 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain1' 
    traindir_mixed0 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain0'
    traindir_mixed1 = r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain1' 
#    mtraindir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain0\*')
#    mtraindir_1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\mtrain1\*') 
    mtraindir_0 = glob.glob(r'C:\Users\atsumilab\Pictures\ds4\mtrain0\*')
    mtraindir_1 = glob.glob(r'C:\Users\atsumilab\Pictures\ds4\mtrain1\*')
    a3d_dir = glob.glob(r'D:\TRIP\Datasets\A3D\frames\*')
#    a3d_sel_dir = glob.glob(r'D:\TRIP\Datasets\A3D\frames\*')
#    a3d_sel_dir = glob.glob(r'D:\TRIP\Datasets\A3D\selected\*')
    a3d_sel_dir = glob.glob(r'E:\TRIP\Datasets\A3D\selected\*')
#    a3d_test_dir = glob.glob(r"D:\TRIP\Datasets\A3D\selected\test_50\*")
    a3d_test_dir = glob.glob(r"E:\TRIP\Datasets\A3D\selected\test_50\*")
#    a3d_test_dir_c = r'D:\TRIP\Datasets\A3D\selected\test_50'
    a3d_test_dir_c = r'E:\TRIP\Datasets\A3D\selected\test_50'
    a3d_use_dir = glob.glob(r"D:\TRIP\Datasets\A3D\selected\non_accident\*")
    a3d_after_100_dir = glob.glob(r"D:\TRIP\Datasets\A3D\selected\others\after_100\*")
    carla_dir = r'E:\TRIP\Datasets\CARLA_dataset\test_3\training'   
    carla_t1_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town01\*") 
    carla_t2_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town02\*") 
    carla_t3_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town03\*") 
    carla_t4_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town04\*") 
    carla_t5_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town05\*") 
    carla_t6_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town06\*") 
    carla_t7_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\Town07\*") 

    carla_glob_dir = glob.glob(r"E:\TRIP\Datasets\CARLA_dataset\test_3\training\*")

#    rename_carla(carla_dir)
    moveTreeFormatA3D(carla_t1_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t2_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t3_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t4_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t5_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t6_dir, carla_glob_dir, "non_accident")
    moveTreeFormatA3D(carla_t7_dir, carla_glob_dir, "non_accident")

#    moveTreeFormatA3D(a3d_test_dir, a3d_sel_dir, "accident")
#    moveTreeFormatA3D(a3d_dir, a3d_sel_dir, folder_name_a3d[0])

#    countFilesFolders(traindir_mixed0, [])
#    countFilesFolders(traindir_mixed1, [])
#    countFilesFolders(traindir_dashcam0, [])
#    countFilesFolders(traindir_dashcam1, [])
#    countFilesFolders(traindir_viena0, [])
#    countFilesFolders(traindir_viena1, [])
    
    #delTreeFormatA3D(a3d_test_dir)
    #delTreeFormatA3D(a3d_test_dir)
    #countFilesFolders(a3d_test_dir_c, [])
    #moveTreeFormatA3D(a3d_use_dir, a3d_sel_dir, "accident")
    #moveTreeFormatA3D(a3d_test_dir, a3d_sel_dir, "accident")
    #moveTreeFormatA3D(a3d_dir, a3d_sel_dir, folder_name_a3d[0])

    #delTreeFormatVirtual(mtraindir_0, Folder.accident_car)
    #delTreeFormatVirtual(mtraindir_0, Folder.accident_asset)
    #delTreeFormatVirtual(mtraindir_0, Folder.accident_pedes)
    #delTreeFormatVirtual(mtraindir_1, Folder.accident_asset)
    #delTreeFormatVirtual(mtraindir_1, Folder.accident_car)
    #delTreeFormatVirtual(mtraindir_1, Folder.accident_pedes)
    #copyTreeFormat(traindir_dashcam, output_dir, Folder.dashcam_train, "train0")
    #moveTreeFormat(traindir_0, output_dir, Folder.dashcam_train, "train1")    
    #countFilesFolders(dashcam_train, Folder.dashcam_train)
    #countFilesFolders(traindir_dashcam0, [])
    #countFilesFolders(traindir_dashcam1, [])
    #countFilesFolders(dashcam_test, Folder.dashcam_test)
    #countFilesFolders(testdir_dashcam0, [])
    #countFilesFolders(testdir_dashcam1, [])
    #countFilesFolders(viena_dir, Folder.accident_asset)
    #countFilesFolders(viena_dir, Folder.accident_car)
    #countFilesFolders(viena_dir, Folder.accident_pedes)
    #countFilesFolders(traindir_viena0, [])
    #countFilesFolders(traindir_viena1, [])
    #countFilesFolders(traindir_mixed0, [])
    #countFilesFolders(traindir_mixed1, [])

    #copyTreeFormat(testdir_dashcam, output_dir, Folder.dashcam_test, "test0")
    #moveTreeFormat(testdir_0, output_dir, Folder.dashcam_test, "test1") 
    #copyTreeFormat(accident_car_dir, output_dir, Folder.accident_car, "vtrain0")
    #copyTreeFormat(accident_asset_dir, output_dir, Folder.accident_asset, "vtrain0")
    #copyTreeFormat(accident_pedestrian_dir, output_dir, Folder.accident_pedes, "vtrain0")
    #moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_asset, "vtrain1")
    #moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_car, "vtrain1")
    #moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_pedes, "vtrain1")

#    for i in range (0, 621):
#        print("Test folfer: " + str(i))
#        copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\test1\\h000" + str(i) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\test1\\000" + str(i) + "_1\\tbox")

#    for j in range (10, 78):
#        print("Train folder: " + str(j))
#        if j < 10:
#            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\bb\\00" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
#            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\img\\00" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
#        else:
#            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\bb\\0" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
#            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\img\\0" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
#        if j < 10:
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h00000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\00000" + str(j) + "_0\\tbox")
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h00000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\00000" + str(j) + "_1\\tbox")
#        elif (j >= 10 and j < 100):
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h0000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\0000" + str(j) + "_0\\tbox")
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h0000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\0000" + str(j) + "_1\\tbox")
#        else:
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\000" + str(j) + "_0\\tbox")
#            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\000" + str(j) + "_1\\tbox")
