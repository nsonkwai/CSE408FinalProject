import numpy as np 
from cv2 import cv2

#single scale retinex to perform retinex algorithm on one channel
def singleRetinex(image, sigma):
    result = np.log10(image) - np.log10(cv2.GaussianBlur(image, (0, 0), sigma))
    return result

#multi scale retinex to iterate through the sigma list to split each channel
def msr(image, sigma_list):
    retinex = np.zeros_like(image)
    for sigma in sigma_list:
        retinex += singleRetinex(image, sigma)
    
    retinex = retinex/len(sigma_list)
    return retinex

#color balance to iterate through the image matrix and take away outlying color values
def colorBalance(image, low, high):
    total = image.shape[0]*image.shape[1]
    for i in range(image.shape[2]):
        unique, counts = np.unique(image[:,:,i], return_counts = True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current)/total < low:
                newLow = u
            if float(current)/total < high:
                newhigh = u
            current += c
        image[:,:,i] = np.maximum(np.minimum(image[:,:,i], newhigh), newLow)
    return image

#color restoration to sum the three separate channels and bring them together
def colorRestoration(image, alpha, beta):
    imageSum = np.sum(image, axis=2, keepdims=True)
    colorRestore = beta * (np.log10(alpha*image)-np.log10(imageSum))
    return colorRestore

#main MSRCR frame for the algorithm to run each function
def multiscaleRet(image, sigma_list, G, b, alpha, beta, low, high):
    image = np.float64(image)+1.0

    image_retinex = msr(image, sigma_list)
    image_color = colorRestoration(image, alpha, beta)
    image_msrcr = G * (image_retinex * image_color + b)
    cv2.imshow('1', image_msrcr)

    for i in range(image_msrcr.shape[2]):
        image_msrcr[:, :, i] = ((image_msrcr[:, :, i] - np.min(image_msrcr[:, :, i])) / (np.max(image_msrcr[:,:,i]) - np.min(image_msrcr[:,:,i]))) * 255

    image_msrcr = np.uint8(np.minimum(np.maximum(image_msrcr, 0), 255))
    cv2.imshow('2', image_msrcr)
    image_msrcr = colorBalance(image_msrcr, low, high)
    cv2.imshow('3', image_msrcr)

    return image_msrcr