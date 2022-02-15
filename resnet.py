# ! pip install tensorflow==2.4.1
# ! pip install pandas
# ! pip install numpy

import os
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf


def get_basename_no_ext(inp):
    return os.path.splitext(os.path.basename(inp[0]))[0]


def get_basenames_no_ext_arr(arr):
    arr_base = np.array(arr).reshape(-1,1)
    arr_base = [os.path.splitext(os.path.basename(inp[0]))[0] for inp in arr_base]
#     arr_base = np.apply_along_axis(get_basename_no_ext, 1, arr_base)
    return arr_base


def check_existance(img_name, true_labels_paths):
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


def get_df(images_path, labels_path):
    img_paths = glob(os.path.join(images_path, "*"))
    label_paths = glob(os.path.join(labels_path, "*"))

    rows = []
    for img_path in img_paths:
        img_class = check_existance(img_path, label_paths)
        rows.append([img_path, img_class])
    return np.array(rows)


def append_images(img_path, req_size):
    img_paths = glob(os.path.join(img_path, "*"))
    dir_size = len(img_paths)
    if dir_size < req_size:
        print("len is less than req_size, appending...")
        for i in range(req_size // dir_size + 1):
            img_paths = img_paths + img_paths
        img_paths = img_paths[:req_size]
    else:
        img_paths = img_paths[:req_size]
    zeros = np.zeros(req_size)
    rows = np.concatenate((np.array(img_paths).reshape(-1, 1), zeros.reshape(-1, 1)), axis=1)
    return rows


train_data = Data("/home/cstar/workspace/create_train_set/data/dataset_out/images/train",
                  "/home/cstar/workspace/create_train_set/data/dataset_out/labels/train")
rows_label_1 = get_df(train_data.image, train_data.label)
img_path_label_0 = "/home/cstar/workspace/create_train_set/data/dataset_out_2/images/train"
rows_label_0 = append_images(img_path_label_0, rows_label_1.shape[0])
rows = np.concatenate((rows_label_0, rows_label_1))
df_train = pd.DataFrame(rows, columns=['filename', 'label'])
df_train.label = df_train.label.astype("float").astype("int8")
# print(df_train.info())
# print(len(df_train[df_train.label == 0]))
# print(len(df_train[df_train.label == 1]))

val_data = Data("/home/cstar/workspace/create_train_set/data/dataset_out/images/train",
                  "/home/cstar/workspace/create_train_set/data/dataset_out/labels/train")
rows_label_1 = get_df(val_data.image, val_data.label)
img_path_label_0 = "/home/cstar/workspace/create_train_set/data/dataset_out_2/images/train"
rows_label_0 = append_images(img_path_label_0, rows_label_1.shape[0])
rows = np.concatenate((rows_label_0, rows_label_1))
df_val = pd.DataFrame(rows, columns=['filename', 'label'])
df_val.label = df_val.label.astype("float").astype("int8")

test_data = Data("/home/cstar/workspace/create_train_set/data/dataset_out/images/train",
                "/home/cstar/workspace/create_train_set/data/dataset_out/labels/train")
rows = get_df(test_data.image, test_data.label)
df_test = pd.DataFrame(rows, columns=['filename', 'label'])
df_test.label = df_val.label.astype("float").astype("int8")

# https://towardsdatascience.com/transfer-learning-for-image-classification-using-tensorflow-71c359b56673