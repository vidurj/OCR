import math
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageChops
import scipy.optimize as optimize
import random
from operator import itemgetter
from scipy import ndimage
import os
import string, cv2
import numpy as np
import codecs
from realData import image_connected_components
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from matplotlib.pyplot import imshow
from realData import centerNonZeroIfPossible

with codecs.open('fontlist.txt','r',encoding='utf8') as handle:
    printable = set(string.printable)
    fontnames = set()
    for line in handle:
        fontfile = line.split(":")[0]
        if fontfile.endswith('.ttf') or fontfile.endswith('.ttc'):
            if len(set(fontfile) - printable) == 0:
                fontnames.add(fontfile.split('/')[-1])

    forbidden = ['Bodoni 72 Smallcaps Book.ttf', 'Copperplate.ttc', 'LastResort.ttf', 'Phosphate.ttc',
                 'Wingdings 2.ttf', 'Wingdings.ttf', "Brush Script.ttf"]

    fontnames = sorted(list(fontnames - set(forbidden)))
    print("found " + str(len(fontnames)) + " fonts on your computer")


# font = ImageFont.truetype("Arial-Bold.ttf",14)
uppercase = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
lowercase = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split()
integers = '0 1 2 3 4 5 6 7 8 9'.split() # ( ) + = - > < ? % ,
degree_symbol = u"\u00B0"
characters = uppercase + lowercase + integers + [degree_symbol]
labeller = dict(zip(characters, range(len(characters))))
labeller["*"] = len(labeller)
words = ["Fig", "to", "awn", "scal", "to", "dra", u"x\u00B0"]
bsChars = "! @ # $ ^ & * / \ | { }".split()


def returnStringOfRandomLength(options, lower, upper):
    length = random.randint(lower, upper)
    return "".join([random.choice(options) for i in xrange(length)])

def fixLength(chars, length):
    return chars + "*" * (length - len(chars))

fonts = []
for name in fontnames:
    try:
        img = Image.new("L", (100, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(name, 6)
        draw.text((0, 0), "\u00B0 a G b ( ) + = - > < .", fill=1, font=font)
        img = np.array(img)
        if np.sum(img) > 0:
            fonts.append(name)
    except:
        pass

# def addCluter(img):
#     pos = (random.randint(0, 28), random.randint(0, 28))
#     pos2 = (pos[0] + random.randint(0, 6),
#             pos[1] + random.randint(0, 6))
#     thickness = random.randint(0, 3)
#     lineType = random.choice([8, 4, 1])
#     radius = random.randint(1, 6)
#     color = 1.0
#     if random.random() < 0.5:
#         cv2.circle(img, pos, radius, color, -thickness, lineType)
#     else:
#         cv2.line(img, pos, pos2, color, thickness, lineType)
#     return img

def getText():
    case = random.randint(0, 5)
    if case == 0:
        return fixLength(returnStringOfRandomLength(characters, 1, 3), 3)
    elif case == 1:
        return random.choice(lowercase) + degree_symbol + "*"
    elif case == 2:
        return random.choice(integers) + random.choice(integers) + degree_symbol
    elif case == 3:
        return random.choice(integers) + random.choice(integers) + "*"
    elif case == 4:
        return random.choice(integers) + random.choice(integers) + random.choice(integers)
    elif case == 5:
        return "***"
    else:
        assert 1 == 0


def getGarbageImage():
    case = random.randint(0, 4)

    poses = (random.randint(0, 28), random.randint(0, 28), random.randint(0, 28), random.randint(0, 28))
    thickness = random.randint(1, 28)
    color = 1.0
    if case == 0:
        return np.zeros((28, 28))
    elif case == 1:
        return np.ones((28, 28))
    elif case == 2:
        img = Image.new("L", (28, 28))
        draw = ImageDraw.Draw(img)
        draw.line(poses)
        return np.array(img)
    elif case == 3:
        img = Image.new("L", (28, 28))
        draw = ImageDraw.Draw(img)
        draw.pieslice(poses, random.random() * 360, random.random() * 360)
        return np.array(img)
    elif case == 4:
        text = returnStringOfRandomLength(bsChars, 1, 3)
        return draw_string(text)
    else:
        assert 1 == 0

def draw_string(text):
    while True:
        img = Image.new("L", (100, 100))
        draw = ImageDraw.Draw(img)
        pos = (50, 50)
        fontName = random.choice(fonts)
        size = random.randint(5, 20)
        font = ImageFont.truetype(fontName, size)
        draw.text(pos, text, fill=1, font=font)
        img = np.array(img)
        if np.sum(img) > 0:
            img = centerNonZeroIfPossible(img, 2, 28, 2, 28)
            if img is not None:
                return img

def generate(batch_size, viz=False):
    data = np.zeros((batch_size, 28, 28))
    labels = np.zeros((batch_size, 3))

    for i in xrange(batch_size):
        text = getText()
        assert len(text) == 3
        labels[i, :] = [labeller[x] for x in text]
        if text == "***":
            img = getGarbageImage()
        else:
            cleanedText = "".join([x for x in text if x != "*"])
            img = draw_string(cleanedText)

        assert np.max(img) <= 1
        assert np.min(img) >= 0
        img = img.astype(float)
        if random.random() < 0.25:
            noise = 2 * np.random.random((28, 28)) - 1
            img += noise / (1 + 9 * random.random())
            mean = np.mean(img)
            img[img < mean] = 0
            img[img >= mean] = 1.0

        if random.random() < 0.5:
            img = np.random.random((28, 28)) * img
        data[i,:,:] = img
        if viz:
            print("_", text, "_")
            # cv2.imshow("afg", img)
            # cv2.waitKey(0)
    assert np.max(data) <= 1
    assert np.min(data) >= 0
    return data, labels


if __name__ == "__main__":
    for i in xrange(10**8):
        data, labels = generate(10000)
        np.save("train_data/" + str(i) + "data", data)
        np.save("train_data/" + str(i) + "labels", labels)
