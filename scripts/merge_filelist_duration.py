from tqdm import tqdm
import json

filelist = open("data/japanese_dataset_char/filelist.txt").read().splitlines()

duration = []
for entry in tqdm(filelist):
    duration.append(float(entry.split("|")[1]))

with open("data/japanese_dataset_char/duration.json", "w") as f:
    json.dump({"duration": duration}, f)
