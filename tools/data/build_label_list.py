import sys
import os


ROOT_DIR = sys.argv[1]
SAVE_DIR = sys.argv[2]
prefix = sys.argv[3]
os.makedirs(SAVE_DIR, exist_ok=True)


for sub_dir in ["train", "test", "valid"]:
    save_file = os.path.join(SAVE_DIR, "{}_{}.txt".format(prefix, sub_dir))
    lines = []
    sub_folder = os.path.join(ROOT_DIR, sub_dir)
    for video_name in os.listdir(sub_folder):
        video_folder = os.path.join(sub_folder, video_name)
        lines.append("{} {} {}".format(video_folder, len(os.listdir(video_folder)), 0))

    with open(save_file, "w") as f:
        f.write("\n".join(lines))
