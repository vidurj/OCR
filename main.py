from model import Network
import os, cv2
import numpy as np
from OCR import labeller, generate
import pylab as plt

revLabeller = dict([(v, k) for k, v in labeller.items()])

def printBestChars(distributions):
    bestOnes = np.argmax(distributions, axis=1)
    return [revLabeller[x] for x in bestOnes]

def getFilesInDir(directory, extension):
    for dirName, subdirList, fileList in os.walk(directory):
        validFiles = [fname for fname in fileList if fname.endswith(extension)]
        images = np.zeros((len(validFiles), 28, 28))
        fileList = set(validFiles)
        for i in range(len(validFiles)):
            data = np.load(directory + "/" + str(i)+extension)
            images[i, :, :] = data# / 255.0
        assert np.max(images) <= 1
        assert np.min(images) >= 0
    return images


test_data = getFilesInDir("test_images", ".npy")

test_data, batch_labels = generate(100)

model = Network(len(test_data))
model.load_weights("next_gen.weights")

results = model.test(test_data)
results = [printBestChars(x) for x in results]
count = 0
for x, y, z in zip(*results):
    print x, y, z
    cv2.imshow("asf", test_data[count, :, :])
    cv2.waitKey(0)
    count+=1


# running_avg = 0
# for i in range(1, 10**8):
#     print(i)
#     loss = model.train()
#     running_avg = 0.99 * running_avg + 0.01 * loss
#     print("avg", running_avg)
#     if i % 1000 == 0:
#         print("Saving")
#         model.save_weights("friday.weights")
#         print("Done")
