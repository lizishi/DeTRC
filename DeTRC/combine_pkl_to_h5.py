import os
import h5py
import numpy as np
import pickle


sub_dir = "train"
ROOT_DIR = "./datasets/UCF/feature-frame-mae/{}".format(sub_dir)

f = h5py.File("./datasets/UCF/feature-frame-mae/{}_rgb.h5".format(sub_dir), "w")
video_list = os.listdir(ROOT_DIR)
for video in video_list:
    video_name = video.replace(".pkl", "")
    with open(os.path.join(ROOT_DIR, video), "rb") as input_file:
        data = pickle.load(input_file)
    f.create_dataset(video_name, data=data.squeeze())
f.close()
