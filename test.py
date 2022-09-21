# import glob
# import os

# image_list = glob.glob("/SPINH/data/H36M/images/*.jpg")
# print(len(image_list))
# for i in image_list:
#     os.rename(i, image_list[1000][:36] + "s" + image_list[1000][36:])

import numpy as np
import cv2
from utils.imutils import crop

data = np.load("data/dataset_extras/3dpw_test.npz")
print(data['imgname'][27900])
img_path = "data/3DPW/" + data['imgname'][27900]
# print(img_path)
img = cv2.imread(img_path)
# center = data['center'][0]
# scale = data['scale'][0]
# img = crop(img, center, scale, (224, 224))
cv2.imwrite('test.jpg', img)