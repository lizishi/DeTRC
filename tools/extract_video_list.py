import os


root_folder = "./datasets/LLSP/video"
result = []
for sub_folder in ["train", "valid", "test"]:
    video_list = os.listdir(os.path.join(root_folder, sub_folder))
    for video in video_list:
        result.append(os.path.join(sub_folder, video))

with open("RepCount.txt", "w") as f:
    f.write("\n".join(result))
