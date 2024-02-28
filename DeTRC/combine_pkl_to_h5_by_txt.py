import os
import h5py
import numpy as np
import pickle


sub_dir = "valid"
ROOT_DIR = "./datasets/LLSP/feature-frame/"
video_list_file = "./datasets/LLSP/temp/annt_file_{}.txt".format(sub_dir)

f = h5py.File("./datasets/LLSP/feature-frame/{}_rgb.h5".format(sub_dir), "w")
with open(video_list_file, "r") as g:
    video_list = g.readlines()

for video_path in video_list:
    video_name = os.path.basename(video_path.split(" ")[0])

    with open(os.path.join(ROOT_DIR, video_name + ".pkl"), "rb") as input_file:
        data = pickle.load(input_file)
    f.create_dataset(video_name, data=data.squeeze())
f.close()
