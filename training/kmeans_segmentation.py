import numpy as np
import cv2
from Gamma import auto_gamma
import math
from joblib import parallel_backend
from joblib import Parallel, delayed
# def normalize(a):
#     return (255*((a - np.min(a))/np.ptp(a)))
def apply_kmeans(img, k = 4):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def check_red(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([165,25,10])
    upper = np.array([180,255,230])
    # red range to right side of HSV scale
    mask = cv2.inRange(image, lower, upper) 
    # red range to left side of HSV scale
    mask2 = cv2.inRange(image, np.array([0,25,10]), np.array([10,255,230])) 
    # combine the two masks to get overall red mask
    mask = cv2.bitwise_or(mask, mask2) 
    # count the number of white pixels in the mask --> these are red color pixels!
    count = np.count_nonzero(mask)
    print('red_count:',count)
    if count > 200:
        return True
    else:
        return False
    

def detector(directory, filename):
    
    img = cv2.imread(os.path.join(directory, filename))
    # make a backup of image for later use for croping
    ors_img = img.copy()
    # resize image to half size for faster computation
    img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
    # adjust gamma of the RGB image
    img = auto_gamma(img)
    img = cv2.GaussianBlur(img,(7,7),3)
    org = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # get kmeans image (apply kmeans on HSV image)
    res = apply_kmeans(img, k = 5)
    
    res2 = res[:,:,2] # choose value channel from HSV
    ret, thresh1 = cv2.threshold(res2, int(res2.max()*0.75), 255, cv2.THRESH_BINARY)
    
    # use elliptical kernel for morphological operation
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion = cv2.erode(closing,kernel,iterations = 7)
    dilation = cv2.dilate(erosion,kernel,iterations = 8)
    

    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea)
    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 9500 < area < 100000:
            contours_area.append(con)
    contours_cirles = []

    # check if contour is of circular shape
    for con in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        
        if 0.7 < circularity < 1.1:
            print (circularity, cv2.contourArea(con))
            contours_cirles.append(con)
    try:
        cv2.drawContours(org, contours_cirles, -1, (0, 255, 0), 3) 
        x,y,w,h = cv2.boundingRect(contours_cirles[-1])
        img = cv2.rectangle(org,(x,y),(x+w,y+h),(0,255,0),2)
        x,y,w,h = (x*2),(y*2),(w*2),(h*2)
        crop_img = ors_img[y:y+h, x:x+w]
        if check_red(crop_img):
            
            crop_img = cv2.resize(crop_img,(256,256))
            cv2.imwrite(os.path.join(r'kmeans\{}'.format(directory), filename),crop_img)
            print(os.path.join(r'kmeans\train', filename))
            return True
        else:
            ors_img = cv2.resize(ors_img,(256,256))
        cv2.imwrite(os.path.join(r'kmeans\{}'.format(directory), filename),ors_img)
        return False
    except:
        ors_img = cv2.resize(ors_img,(256,256))
        cv2.imwrite(os.path.join(r'kmeans\{}'.format(directory), filename),ors_img)
        return False

    


import os
import pandas as pd

directory = 'images/train'
files_train = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        files_train.append([directory, filename])

directory = 'images/test'
files_test = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        files_test.append([directory, filename])
        
def run(filess):
    modified = detector(filess[0], filess[1])
    if modified:
        return [filename, 1]
    else:
        return [filename, 0]


with parallel_backend('loky', n_jobs=10):
    modified_list = Parallel()(delayed(run)(f) for f in files_train)
df = pd.DataFrame(modified_list)
df.to_csv('train_modrec.csv')

with parallel_backend('loky', n_jobs=9):
    modified_list = Parallel()(delayed(run)(f) for f in files_test)
df = pd.DataFrame(modified_list)
df.to_csv('test_modrec.csv')