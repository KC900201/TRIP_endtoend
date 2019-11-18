"""
Created on Sat Sep 14 13:05:58 2019

@author: atsumilab
@filename: csvtotxt.py
@code: utf-8
========================
Date          Comment
========================
11182019      First revision
"""
import csv
import os

if __name__ == '__main__':
    path = input('Enter the path: ')

    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'):
                extension = ('.txt')
                print(os.path.join(path, filename))
                new_txt_file = str(os.path.splitext(filename)[0] + extension)
               
                try:
                    my_input_file = open(os.path.join(path, filename), 'r')
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))

                if not my_input_file.closed:
                    text_list = [];
                    for line in my_input_file.readlines():
                        line = line.split(",", 2)
                        text_list.append(" ".join(line))
                    my_input_file.close()

                try:
                    my_output_file = open(os.path.join(path, new_txt_file), 'w+')
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))

                if not my_output_file.closed:
                    my_output_file.write("#1\n")
                    my_output_file.write("double({},{})\n".format(len(text_list), 2))
                    for line in text_list:
                        my_output_file.write("  " + line)
                    print('File Successfully written.')
                    my_output_file.close()

