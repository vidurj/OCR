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
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from matplotlib.pyplot import imshow

with codecs.open('fontlist.txt','r',encoding='utf8') as handle:
    printable = set(string.printable)
    fontnames = set()
    for line in handle:
        fontfile = line.split(":")[0]
        if fontfile.endswith('.ttf') or fontfile.endswith('.ttc'):
            if len(set(fontfile) - printable) == 0:
                fontnames.add(fontfile.split('/')[-1])

    forbidden = ['Bodoni 72 Smallcaps Book.ttf', 'Copperplate.ttc', 'LastResort.ttf', 'Phosphate.ttc',
                 'Wingdings 2.ttf', 'Wingdings.ttf']

    fontnames = sorted(list(fontnames - set(forbidden)))
    print("found " + str(len(fontnames)) + " fonts on your computer")


# font = ImageFont.truetype("Arial-Bold.ttf",14)
uppercase = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
lowercase = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split()
integers = '0 1 2 3 4 5 6 7 8 9'.split()
degree_symbol = u"\u00B0"
characters = uppercase + lowercase + integers + [degree_symbol]
labeller = dict(zip(characters, range(len(characters))))
labeller["*"] = len(labeller)
words = ["Fig", "to", "awn", "scal", "to", "dra", u"x\u00B0"]


fonts = []
for name in fontnames:
    try:
        ImageFont.truetype(name, 24)
        fonts.append(name)
    except:
        pass


def addCluter(img):
    pos = (random.randint(0, 28), random.randint(0, 28))
    pos2 = (pos[0] + random.randint(0, 6),
            pos[1] + random.randint(0, 6))
    thickness = random.randint(0, 3)
    lineType = random.choice([8, 4, 1])
    radius = random.randint(1, 6)
    color = 1.0
    if random.random() < 0.5:
        cv2.circle(img, pos, radius, color, -thickness, lineType)
    else:
        cv2.line(img, pos, pos2, color, thickness, lineType)
    return img

def getText():
    case = random.randint(0, 9)
    if case == 0:
        return random.choice(characters) + "**"
    elif case == 1:
        return random.choice(characters) + random.choice(characters) + "*"
    elif case == 2:
        return random.choice(characters) + random.choice(characters) + random.choice(characters)
    elif case == 3:
        return random.choice(lowercase) + degree_symbol + "*"
    elif case == 4:
        return random.choice(integers) + random.choice(integers) + degree_symbol
    elif case == 5:
        return random.choice(integers) + random.choice(integers) + "*"
    elif case == 6:
        return random.choice(integers) + random.choice(integers) + random.choice(integers)
    elif case == 7:
        return random.choice(lowercase) + "**"
    elif case == 8:
        return random.choice(uppercase) + "**"
    else:
        return "***"


def getGarbageImage():
    case = random.random()
    base = np.zeros((28, 28))
    pos = (random.randint(0, 28), random.randint(0, 28))
    pos2 = (random.randint(0, 28), random.randint(0, 28))
    thickness = random.randint(0, 10)
    lineType = random.choice([8, 4, 1])
    radius = random.randint(1, 20)
    color = 1.0
    if case < 0.1:
        base += 1
    if case < 0.33:
        cv2.circle(base, pos, radius, color, thickness * random.choice([1, -1]), lineType)
    elif case < 0.66:
        cv2.line(base, pos, pos2, color, thickness, lineType)
    elif case < 0.9:
        cv2.arrowedLine(base, pos, pos2, color, thickness, lineType)
    return base



def generate(batch_size, viz=False):
    height, width = 50, 50
    data = np.zeros((batch_size, 28, 28))
    labels = np.zeros((batch_size, 3))
    for i in range(batch_size):

        text = getText()
        assert len(text) == 3
        labels[i, :] = [labeller[x] for x in text]
        text = "".join([x for x in text if x!="*"])
        if text == "***":
            im = getGarbageImage()
        else:
            img = Image.new("L", (100, 100))#(random.randint(50, 200), random.randint(50, 200)))
            draw = ImageDraw.Draw(img)
            pos = (0, 0)#(random.randint(0, 100), random.randint(0, 100))
            font = ImageFont.truetype(random.choice(fonts), random.randint(6, 20))
            draw.text(pos, text, fill=1, font=font)
            im = img.crop(img.getbbox())
            im = np.array(im.resize((28, 28))) / 1.0

        assert np.max(im) <= 1
        assert np.min(im) >= 0
        noise = 2*np.random.random((28, 28)) - 1
        #im = addCluter(im)
        if random.random() < 0.3:
            #print "Adding noise"
            im += noise / 8.0
        mean = np.mean(im)
        im[im < mean] = 0
        im[im >= mean] = 1.0
        if random.random() < 0.4:
            #print "smoothing"
            im = np.random.random((28, 28)) * im
        data[i,:,:] = im
        if viz:
            print "_", text, "_"
            cv2.imshow("afg", im)
            cv2.waitKey(0)
    assert np.max(data) <= 1
    assert np.min(data) >= 0
    return data, labels


if __name__ == "__main__":
    print generate(1000, viz=True)
