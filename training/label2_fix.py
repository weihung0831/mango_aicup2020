
# -*- coding:utf-8 -*-
import pandas as pd 
import numpy as np 
import cv2 as cv 
import math
import ntpath
import matplotlib.pyplot as plt 
import os

from pathlib import Path
# read the csv
df = pd.read_csv('dev.csv', encoding= "utf-8", header=None,low_memory=False)

j = 0
# initial buffer for the whole csv
defect_types = []
# iterate over rows of the csv
def create_directory(path):
    if not os.path.exists(path):
        try:
            path = Path(path)
            path.mkdir(parents = True)
        except:
            print("An error encountered while creating directories.")
            

            
namecount = 0
buffer = []
for index, row in df.iterrows():
    # print(index, row.values)
    # make a buffer list to collect string data
    stringlist = []
    box_locations = []
    # iterate over the element of rows (to convert pandas to numpy = .values)
    for data in row.values:
        # if the type of data is 'str' then add to the buffer
        if type(data) is str:
            stringlist.append(data)
        else:
            if not math.isnan(data):
                box_locations.append(int(data))
                
    box_locations = np.array(box_locations).reshape(-1,4)
    
    # print(stringlist, end = '\r')
    labels_ = [0,0,0,0,0]
    
    for count in range(len(box_locations)):
        
        
        strg = stringlist[count+1]
        
        if strg == r'不良-乳汁吸附':
            labels_[0] = 1
        elif strg == r'不良-機械傷害':
            labels_[1] = 1
        elif strg==r'不良-炭疽病':
            labels_[2] = 1
        elif strg ==r'不良-著色不佳':
            labels_[3] = 1
        elif strg == r'不良-黑斑病':
            labels_[4] = 1
        
        # print(save_loc)
        
        
        namecount+=1
    # print(labels_)
    buffer.append(['/content/images/test/'+stringlist[0]]+labels_)
# print(buffer)
new_lab = pd.DataFrame(buffer)
new_lab.to_csv('test_labelsx.csv')
