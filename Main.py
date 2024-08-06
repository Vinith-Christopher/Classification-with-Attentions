
# <-------- Import Necessary Packages ------->
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from termcolor import cprint


# dataset from my folder
path = 'DATASET/natural_images'
labels = list()
Images = list()
for folder in os.listdir(path):
    for image in os.listdir(path + '/' +folder ):
        image = path + '/' +folder + '/' + image
        img = cv2.imread(image)   # read
      
        # Create the sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Sharpen the image
        sharpened_image = cv2.filter2D(img, -1, kernel)

        resized_image = cv2.resize(sharpened_image, (250, 250)) # resize

        cprint('Preprocessing Complete for the ---- ' + image , 'green')
        Images.append(resized_image)
        labels.append(folder)

x = np.array(Images)
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

print(x.shape)
print(y.shape)



