# ! pip install tensorflow==2.4.1
# ! pip install pandas
# ! pip install numpy

import os
from glob import glob

import numpy as np
import cv2 as cv

path_in = "/home/cstar/workspace/data/architecture_and_3d-printers"
path_out = "/home/cstar/workspace/data/architecture_and_3d-printers_bw"

imgs_in = glob(os.path.join(path_in, "*"))
for i, img_in in enumerate(imgs_in):
    print(i, img_in)
    if os.path.splitext(img_in)[1] == ".gif":
        continue
    img = cv.imread(img_in)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = [img, img, img]
    img = np.transpose(img, (1, 2, 0))
    path_out_img = os.path.join(path_out, os.path.basename(img_in))
    cv.imwrite(path_out_img, img)
