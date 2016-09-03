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
print "There are " + str(len(geoimages)) + " images in the Kaplan Geometry set."

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

global count
count = 0


def sendToOCR():
    results = []
    address = "http://vision-ocr.dev.allenai.org/v1/ocr"
    for i in ["own", "new"] + range(2215):
        encoded_string = base64.b64encode(open("test_images/"+str(i)+".png", "rb").read())
        r = requests.post(address, data={"image": encoded_string})
        print json.loads(r.text)
        print [x["value"] for x in json.loads(r.text)["detections"]]

#sendToOCR()

def get_small_subimages(subimages, width_lower_bd, width_upper_bd, height_lower_bd, height_upper_bd):
    small_subimages = []
    for img in subimages:
        nonzero = np.argwhere(img)
        xes = [x[0] for x in nonzero]
        xstart = min(xes)
        xstop = max(xes)
        yes = [x[1] for x in nonzero]
        ystart = min(yes)
        ystop = max(yes)
        height, width = ystop - ystart, xstop - xstart
        if height > height_lower_bd and height < height_upper_bd and width > width_lower_bd and width < width_upper_bd:
            centered_img = np.zeros((width_upper_bd, height_upper_bd))
            extra_x = (width_upper_bd - width) / 2.0
            extra_y = (height_upper_bd - height) / 2.0

            centered_img[extra_x: extra_x + width, extra_y: extra_y + height] = img[xstart:xstop, ystart:ystop]
            small_subimages.append(centered_img)
            np.save("test_images/" + str(count), centered_img)
            # plt.imshow(centered_img)
            # plt.title("b")
            # plt.show()
            #cv2.imwrite("test_images/" + str(count)+".png", centered_img)
            global count
            count += 1
    return small_subimages



# pieces = []
# label = 0
# answers = []
for img_file in geoimages:
    img = 1 - cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2GRAY) / 255.0
    # cv2.imshow("fsg", img)
    # cv2.waitKey(0)
    assert np.median(img) < 0.5
    print "median color", np.median(img), "shape", img.shape
    components = image_connected_components(img)
    subimages = get_small_subimages(components, 2, 28, 2, 28)
#     orig_img = cv2.imread(img_file).copy()
#     output = cv2.imread(img_file).copy()
    # for bbox in subimages:
    #     # cv2.rectangle(output, bbox[:2], bbox[2:], (0, 128, 255), -1)
    #     x1, y1, x2, y2 = bbox
    #     var = img_bw[y1:y2, x1:x2]
    #     var = cv2.resize(var, (28, 28))
    #     cv2.imwrite("test_images/" + str(label) + ".png", var)
    #     label += 1
    #     if label < len(answers):
    #         continue
    #     # encoded_string = base64.encodestring(open("temp.png", "rb").read())
    #     #         #print pieces[-1]
    #     #         r = requests.post(address, data={"image": encoded_string})
    #     #         print r
    #     #         print json.loads(r.text)
    #     plt.subplot(1, 2, 1), plt.imshow(orig_img, 'gray')
    #     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(1, 2, 2), plt.imshow(var)
    #     plt.title('Identified Text'), plt.xticks([]), plt.yticks([])
    #     plt.show()
    #     cv2.imshow("fdg", orig_img)
    #     cv2.waitKey(0)
