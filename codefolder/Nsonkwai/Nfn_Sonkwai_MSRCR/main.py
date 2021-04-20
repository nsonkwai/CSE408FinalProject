import os
import sys
from cv2 import cv2
import MSRCR
import time

#Time data collection
start_time = time.time()

#arguments for MSRCR algorithm
sigma_list = [15, 80, 250]
G = 5.0
b = 25.0
alpha = 125.0
beta = 46.0
low = 0.01
high = 0.99

#Get images from data folder
data_path = 'data'
img_list = os.listdir(data_path)
if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

#Iterate through each image to perform MSRCR on each
for imageName in img_list:
    print('here')
    if imageName == '.gitkeep':
        continue

    image = cv2.imread(os.path.join(data_path, imageName))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_msrcr = MSRCR.multiscaleRet(image, sigma_list, G, b, alpha, beta, low, high)

    shape = image.shape
    cv2.imshow('Original', image)
    cv2.imshow('MSRCR', image_msrcr)
    cv2.waitKey(0)

print("Process finished --- %s seconds ---" % (time.time() - start_time))
