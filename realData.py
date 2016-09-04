import math
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.optimize as optimize
import random
from operator import itemgetter
from PIL import ImageChops
import matplotlib.cm as cm
from scipy import ndimage
import cv2
import os
import base64
import requests
import json
import os
import io
import PIL
from PIL import Image
import glob


def get_image_files(directory, extension=".png"):
    files = [f for f in os.listdir(directory)]
    return [directory + '/' + f for f in files if f.endswith(extension)]


geoimages = get_image_files('./geoimages', '.png')
print("There are " + str(len(geoimages)) + " images in the Kaplan Geometry set.")

def image_connected_components(img, connectivity = 'diagonal'):
    def flat_for(a, f):
        from copy import copy, deepcopy
        b = deepcopy(a)
        b.reshape(-1)
        for i, v in enumerate(b):
            b[i] = f(v)
        return b
    if connectivity == 'diagonal':
        s = [[1,1,1],
             [1,1,1],
             [1,1,1]]
        label_im, nb_labels = ndimage.label(img, structure=s)
    else:
        label_im, nb_labels = ndimage.label(img)
    results = [flat_for(label_im, lambda x: (x == label+1) * 1) for label in range(nb_labels)]
    return results




def sendToOCR():
    results = []
    address = "http://vision-ocr.dev.allenai.org/v1/ocr"
    for i in ["own", "new"] + range(2215):
        encoded_string = base64.b64encode(open("test_images/"+str(i)+".png", "rb").read())
        r = requests.post(address, data={"image": encoded_string})
        print(json.loads(r.text))
        print([x["value"] for x in json.loads(r.text)["detections"]])

#sendToOCR()

def relu(x):
    return max(x, 0)

def centerNonZeroIfPossible(regionOfInterest, width_lower_bd, width_upper_bd,
                            height_lower_bd, height_upper_bd, origImage=None):
    if origImage is None:
        origImage = regionOfInterest
    nonzero = np.argwhere(regionOfInterest)
    xstart = min(nonzero, key=lambda x: x[0])[0]
    xstop = max(nonzero, key=lambda x: x[0])[0] + 1
    ystart = min(nonzero, key=lambda x: x[1])[1]
    ystop = max(nonzero, key=lambda x: x[1])[1] + 1
    height, width = ystop - ystart, xstop - xstart

    if height > height_lower_bd and height < height_upper_bd and width > width_lower_bd and width < width_upper_bd:
        centered_img = np.zeros((width_upper_bd, height_upper_bd))
        extra_x = (width_upper_bd - width) / 2.0
        xstart -= int(math.floor(extra_x))
        centered_x_start = relu(- xstart)
        xstart = relu(xstart)
        xstop += int(math.ceil(extra_x))
        centered_x_stop = width_upper_bd - relu(xstop - origImage.shape[0])
        xstop = min(xstop, origImage.shape[0])

        extra_y = (height_upper_bd - height) / 2.0
        ystart -= int(math.floor(extra_y))
        centered_y_start = relu(- ystart)
        ystart = relu(ystart)
        ystop += int(math.ceil(extra_y))
        centered_y_stop = height_upper_bd - relu(ystop - origImage.shape[1])
        ystop = min(ystop, origImage.shape[1])
        centered_img[centered_x_start: centered_x_stop, centered_y_start:centered_y_stop] = \
            origImage[xstart:xstop, ystart:ystop]
        return centered_img
    else:
        return None

global count
count = 0

def get_small_subimages(origImage, subimages, width_lower_bd, width_upper_bd, height_lower_bd, height_upper_bd):
    small_subimages = []
    for img in subimages:
        centered_img = centerNonZeroIfPossible(img, width_lower_bd, width_upper_bd, height_lower_bd, height_upper_bd)
        if centered_img is not None:
            small_subimages.append(centered_img)
            # cv2.imshow("fg", centered_img)
            # cv2.waitKey(0)
            np.save("test_images/" + str(count), centered_img)
            global count
            count += 1
    return small_subimages


if __name__ == "__main__":
    # pieces = []
    # label = 0
    # answers = []
    for img_file in geoimages:
        img = 1 - cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY) / 255.0
        # cv2.imshow("fsg", img)
        # cv2.waitKey(0)
        assert np.median(img) < 0.5
        print("median color", np.median(img), "shape", img.shape)
        components = image_connected_components(img)
        subimages = get_small_subimages(img, components, 2, 28, 2, 28)