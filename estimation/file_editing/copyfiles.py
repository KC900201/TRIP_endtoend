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
"""

import shutil
import errno 
import os
import stat
import glob

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

if __name__ == '__main__':
    #11292019
    testdir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\test0\*')
    testdir1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\test1\*')
    traindir = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\train0\*')
    traindir1 = glob.glob(r'E:\AtsumiLabMDS-2\TRIP\Trip2018Q1\Dashcam\ds3\train1\*')
    
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
    
    #for i in range (456, 621):
        #print("Test folfer: " + str(i))
        #copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\test1\\h000" + str(i) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\test1\\000" + str(i) + "_1\\tbox")
    
    '''
    for j in range (10, 78):
        print("Train folder: " + str(j))
        if j < 10:
            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\bb\\00" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\img\\00" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
        else:
            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\bb\\0" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
            copyTree("D:\\darknet_project\\Dataset\\GTAV\\images\\img\\0" + str(j), "D:\\darknet_project\\Dataset\\GTAV\\images")
        if j < 10:
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h00000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\00000" + str(j) + "_0\\tbox")
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h00000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\00000" + str(j) + "_1\\tbox")
        elif (j >= 10 and j < 100):
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h0000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\0000" + str(j) + "_0\\tbox")
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h0000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\0000" + str(j) + "_1\\tbox")
        else:
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train0\\h000" + str(j) + "_0\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train0\\000" + str(j) + "_0\\tbox")
            copy("E:\\AtsumiLabMDS-2\TRIP\\Trip2018Q1\Dashcam\\ds2\\train1\\h000" + str(j) + "_1\\tbox", "E:\\AtsumiLabMDS-2\\TRIP\\Trip2018Q1\\Dashcam\\ds3\\train1\\000" + str(j) + "_1\\tbox")
    '''