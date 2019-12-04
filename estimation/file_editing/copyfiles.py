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
"""

import shutil
import errno 
import os
import stat
import glob
import math

from folder import Folder

folder_name = ['conv33', 'conv39', 'conv45', 'ebox', 'img', 'orig_img']

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
            fileno = int(filename)
            if fileno > 50:
                new_fileno = fileno - 50
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno == 50) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item).replace(filename, new_filename) 
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
                d = os.path.join(dst, item).replace(filename, new_filename)
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

# 12042019
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
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno == 75) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item).replace(filename, new_filename) 
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
                new_filename = filename.replace(str(fileno), "0" + str(new_fileno)) if (new_fileno < 10 or new_fileno == 75) else filename.replace(str(fileno), str(new_fileno))
                s = os.path.join(src, item)
                d = os.path.join(dst, item).replace(filename, new_filename)
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

def copyTreeFormat(src, dst, list, oname):
    for dir in src:
        if os.path.isdir(dir):
            file_name = os.path.basename(dir)
            print("Folder: " + str(dir))
            if file_name in list:
                for odir in dst:
                    if os.path.isdir(odir):
                        if odir.endswith(oname):
#                            new_odir = odir + "\\" + file_name + "_0" # for dashcam data
                            new_odir = odir + "\\" + file_name  # for virtual data
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
                        if odir.endswith(oname):
                            new_odir = odir + "\\" + file_name.replace("_0", "_1")
                            moveTree(dir, new_odir)

# 12042019
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
    testdir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\test0\*')
    vtraindir_0 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds4\vtrain0\*')

    #copyTreeFormat(traindir_dashcam, output_dir, Folder.dashcam_train, "train0")
    #copyTreeFormat(testdir_dashcam, output_dir, Folder.dashcam_test, "test0")
    #moveTreeFormat(traindir_0, output_dir, Folder.dashcam_train, "train1")
    #moveTreeFormat(testdir_0, output_dir, Folder.dashcam_test, "test1") 
    copyTreeFormat(accident_car_dir, output_dir, Folder.accident_car, "vtrain0")
    copyTreeFormat(accident_asset_dir, output_dir, Folder.accident_asset, "vtrain0")
    copyTreeFormat(accident_pedestrian_dir, output_dir, Folder.accident_pedes, "vtrain0")
    moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_asset, "vtrain1")
    moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_car, "vtrain1")
    moveTreeFormatVirtual(vtraindir_0, output_dir, Folder.accident_pedes, "vtrain1")

    '''
    for dir in no_accident_dir:
        if os.path.isdir(dir):
            print("No accident folder: " + str(dir))
        for odir in output_dir:
            if os.path.isdir(odir):
                if odir.endswith("vtrain0"):
                    file_name = os.path.basename(dir)
                    if file_name in Folder.non_accident:
                        new_odir = odir + "\\" + file_name
                        copyTree(dir, new_odir)        

    for dir in accident_car_dir:
        if os.path.isdir(dir):
            print("Accident car folder: " + str(dir))
        for odir in output_dir:
            if os.path.isdir(odir):
                if odir.endswith("vtrain1"):
                    file_name = os.path.basename(dir)
                    if file_name in Folder.accident_car:
                        new_odir = odir + "\\" + file_name
                        copyTree(dir, new_odir)  

    for dir in accident_asset_dir:
        if os.path.isdir(dir):
            print("Accident asset folder: " + str(dir))
        for odir in output_dir:
            if os.path.isdir(odir):
                if odir.endswith("vtrain1"):
                   file_name = os.path.basename(dir)
                   if file_name in Folder.accident_asset:
                      new_odir = odir + "\\" + file_name
                      copyTree(dir, new_odir)  
    
    for dir in accident_pedestrian_dir:
        if os.path.isdir(dir):
            print("Accident pedestrian folder: " + str(dir))
        for odir in output_dir:
            if os.path.isdir(odir):
                if odir.endswith("vtrain1"):
                   file_name = os.path.basename(dir)
                   if file_name in Folder.accident_pedes:
                    new_odir = odir + "\\" + file_name
                    copyTree(dir, new_odir)  
    
    for dir in accident_b_dir:
        if os.path.isdir(dir):
            print("Accident b folder: " + str(dir))
        for odir in output_dir:
            if os.path.isdir(odir):
                if odir.endswith("vtrain1"):
                    file_name = os.path.basename(dir)
                    new_odir = odir + "\\" + file_name
                    copyTree(dir, new_odir)  
    
    for dir in testdir:
        if os.path.isdir(dir):
            print("Test folder 0: " + str(dir))
            input_dir = dir + "\\orig_img"
            output_dir = input_dir.replace('ds3', 'ds4')
            input_dir_tbox = dir + "\\tbox"
            output_dir_tbox = input_dir_tbox.replace('ds3', 'ds4')
            copyTree(input_dir, output_dir)
            copyTree(input_dir_tbox, output_dir_tbox)

    for dir in traindir:
        if os.path.isdir(dir):
            print("Train folder 0: " + str(dir))
            input_dir = dir + "\\orig_img"
            output_dir = input_dir.replace('ds3', 'ds4')
            input_dir_tbox = dir + "\\tbox"
            output_dir_tbox = input_dir_tbox.replace('ds3', 'ds4')
            copyTree(input_dir, output_dir)
            copyTree(input_dir_tbox, output_dir_tbox)

    for dir in testdir1:
        if os.path.isdir(dir):
            print("Test folder 1: " + str(dir))
            input_dir = dir + "\\orig_img"
            output_dir = input_dir.replace('ds3', 'ds4')
            input_dir_tbox = dir + "\\tbox"
            output_dir_tbox = input_dir_tbox.replace('ds3', 'ds4')
            copyTree(input_dir, output_dir)
            copyTree(input_dir_tbox, output_dir_tbox)

    for dir in traindir1:
        if os.path.isdir(dir):
            print("Train folder 1: " + str(dir))
            input_dir = dir + "\\orig_img"
            output_dir = input_dir.replace('ds3', 'ds4')
            input_dir_tbox = dir + "\\tbox"
            output_dir_tbox = input_dir_tbox.replace('ds3', 'ds4')
            copyTree(input_dir, output_dir)
            copyTree(input_dir_tbox, output_dir_tbox)
     '''
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
