from math import pi
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from numpy import asarray 
import os 
import cv2
from PIL import Image as im 
import numpy as np
import matplotlib.pyplot as plt
path = './data/test/' 
pathAu = './data/augmentation' 
IMAGE_SIZE = 48
batch_size = 128
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    index=0
    for img, ax in zip( images_arr, axes):
        plt.imsave(pathAu+str(index)+".png", img)
        print(img)
        index=index+1

def rotate_images(path):
    image_generator = ImageDataGenerator(rescale=1./255,rotation_range=135)
    data_generator = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    augmented_images = [data_generator[0][0][0] for i in range(5)]
    plotImages(augmented_images)
	

rotate_images(path)







