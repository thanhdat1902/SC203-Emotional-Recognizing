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
path = './data/need' 
pathAu = './data/augmentation/' 
IMAGE_SIZE = 48
batch_size = 3829
def plotImages(images_arr, index):
    for img in images_arr:
        plt.imsave(pathAu+str(index)+".png", img)
        index=index+1

def rotate_images(path):
    image_generator = ImageDataGenerator(rescale=1./255,rotation_range=135)
    data_generator = image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=path,
    shuffle=True,
    target_size=(IMAGE_SIZE, IMAGE_SIZE)
    )
    return data_generator
	

data_generator= rotate_images(path)
index = 0    
num_rotate = 1
for k in range(batch_size):
    augmented_images = [data_generator[0][0][k] for i in range(num_rotate)]
    plotImages(augmented_images, index)
    index = index + num_rotate






