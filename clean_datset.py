import os
import sys

list_path = "data/coco/5k.txt"
output_file_path = "data/coco/5k_modified.txt"
img_files = None
with open(list_path, "r") as file:
    img_files = file.readlines()

label_files = [
    path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
    for path in img_files
]


files_to_keep = []
for i, f in enumerate(label_files):
    f = f.rstrip()
    if os.path.exists(f):
        files_to_keep.append(img_files[i])
    else:
        print(f)
with open(output_file_path, 'w') as filehandle:
    filehandle.writelines("%s\n" % img.rstrip() for img in files_to_keep)
