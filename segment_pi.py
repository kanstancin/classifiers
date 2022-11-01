import cv2 as cv
from glob import glob

import os
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def get_basename_no_ext(inp):
    return os.path.splitext(os.path.basename(inp[0]))[0]


def get_basenames_no_ext_arr(arr):
    arr_base = np.array(arr).reshape(-1,1)
    arr_base = [os.path.splitext(os.path.basename(inp[0]))[0] for inp in arr_base]
#     arr_base = np.apply_along_axis(get_basename_no_ext, 1, arr_base)
    return arr_base


def check_existance(img_name, true_labels_paths):
#     true_labels_paths = glob(os.path.join(inp_path, "*"))
    true_labels_paths_base = get_basenames_no_ext_arr(true_labels_paths)
    img_name_no_ext = os.path.splitext(os.path.basename(img_name))[0]
    if img_name_no_ext in true_labels_paths_base:
        return 1
    else:
        return 0


@dataclass
class Data:
    image: str
    label: str
    mask: str=""


def get_df(images_path, labels_path):
    img_paths = glob(os.path.join(images_path, "*"))
    label_paths = glob(os.path.join(labels_path, "*"))

    rows = []
    for i, img_path in enumerate(img_paths):
        img_class = check_existance(img_path, label_paths)
        rows.append([img_path, img_class, 0])
    return np.array(rows)

def get_df_acc(images_path, labels_path, masks_path):
    img_paths = glob(os.path.join(images_path, "*"))
    mask_paths = [os.path.join(masks_path, f"mask{os.path.splitext(os.path.basename(img_path))[0][3:]}.jpg") for img_path in img_paths]
    zeros = np.ones(len(img_paths))
    rows = np.concatenate((np.array(img_paths).reshape(-1, 1), zeros.reshape(-1, 1),
                           np.array(mask_paths).reshape(-1, 1)), axis=1)
    return rows

def get_df_real(images_path, masks_path):
    img_paths = glob(os.path.join(images_path, "*"))
    mask_paths = [os.path.join(masks_path, f"{os.path.splitext(os.path.basename(img_path))[0]}.png") for img_path in img_paths]
    zeros = np.ones(len(img_paths))
    rows = np.concatenate((np.array(img_paths).reshape(-1, 1), zeros.reshape(-1, 1),
                           np.array(mask_paths).reshape(-1, 1)), axis=1)
    return rows

def append_images(img_path, req_size):
    img_paths = glob(os.path.join(img_path, "*"))
    dir_size = len(img_paths)
    print(dir_size)
    if dir_size < req_size:
        print("len is less than req_size, appending...")
        for i in range(req_size // dir_size + 1):
            img_paths = img_paths + img_paths
        img_paths = img_paths[:req_size]
    else:
        img_paths = img_paths[:req_size]
    zeros = np.zeros(req_size)
    rows = np.concatenate((np.array(img_paths).reshape(-1, 1), zeros.reshape(-1, 1),
                           zeros.reshape(-1, 1)), axis=1)
    return rows
    
    
test_data = Data("/home/cstar/workspace/img/",
                "1")
rows = get_df(test_data.image, test_data.label)
print(rows)
df_test = pd.DataFrame(rows, columns=['filename', 'label', 'mask_'])
df_test.label = df_test.label.astype("float").astype("int8")

import os

img_size = (720, 720)
num_classes = 2
batch_size = 1


#import imgaug.augmenters as iaa

# process mask
def resize_mask(mask, target_shape):
    kernel = np.ones((15, 15), np.uint8)
    mask =  cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
    mask =  cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    shape = (target_shape[1], target_shape[0])
    mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST) #cv.INTER_AREA
    return mask

def img_to_binary(mask):
    mask[mask<=125] = 0
    mask[mask>125] = 1
    return mask

def img_to_binary(mask):
    mask[mask<=125] = 0
    mask[mask>125] = 1
    return mask

#from imgaug.augmentables.segmaps import SegmentationMapsOnImage



def preprop_img(img, target_size):
    shape = (target_size[1], target_size[0])
    img = cv.resize(img, shape, interpolation=cv.INTER_AREA)
    return img

def preprop_mask(mask, target_size):
    shape = (target_size[1], target_size[0])
    mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)
    mask = img_to_binary(mask)
    return mask
    
def augment(img, mask, seq, shape):
    mask = SegmentationMapsOnImage(mask, shape=shape)
    img, mask = seq(image=img, segmentation_maps=mask)
    return img, mask.get_arr()
    
def load_data(inp_path, target_path, shape=(1000, 1000)):
    img = cv.imread(inp_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, shape, interpolation=cv.INTER_AREA)
    mask = cv.imread(target_path, cv.IMREAD_GRAYSCALE)
    kernel = np.ones((15, 15), np.uint8)
    mask =  cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
    mask =  cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)
    mask[mask<=30] = 0
    mask[mask>30] = 255
    # img, mask = augment(img, mask, seq, shape)
    return img, mask
    
def load_img_cv(path, target_size, is_mask=False, augment=True):
    if not is_mask:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        shape = (target_size[1], target_size[0])
        img = cv.resize(img, shape, interpolation=cv.INTER_AREA)
        if augment:
            aug = iaa.GammaContrast((0.5, 3.0))
            img = aug(image=img)
    else:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img[img<=30] = 0
        img[img>30] = 255
        img = resize_mask(img, target_size)
        img = img_to_binary(img)
    return img

class OxfordPets2(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, target_size, input_img_paths, target_img_paths, augment=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.target_size = target_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.augment = augment

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.target_size + (1,), dtype="uint8")
        for i, paths in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            path_input, path_target = paths
            img, mask = load_data(path_input, path_target)
            img = preprop_img(img, self.img_size)
            mask = preprop_mask(mask, self.target_size)
            x[i] = np.expand_dims(img, 2)
            y[i] = np.expand_dims(mask, 2)
        return x, y


from tensorflow.keras.models import load_model

# Load the previously saved weights
model = load_model("seg_model/seg_only_sim.h5")

# test run
test_input_img_paths = df_test.filename.to_list()
# val_target_img_paths = df_test.mask_.astype(str).to_list()

target_size = (90, 90)

# Instantiate data Sequences for each split
test_gen = OxfordPets2(1, img_size, target_size, test_input_img_paths, test_input_img_paths, 
                      augment=False)
print("st")
test_preds = model.predict(test_gen)
print("end")

