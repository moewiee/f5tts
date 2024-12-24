# We use TXT since it take less RAM.

from time import time
import json
import os
from mutagen import File
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from model.utils import convert_char_to_pinyin, convert_char_to_charlist
import uuid

PROCESSING_JAPANESE = True

txt_files = [
    # '/mnt/weka/tts/jchat_f5/content.txt',
    '/mnt/md0/tts/data/jaudiobook.txt',
    '/mnt/md0/tts/data/jemilia/content.txt',
    '/mnt/md0/tts/data/jreazon.txt',
]

out_dir = 'data/japanese_dataset_char'

def prepare_folder_dict(metadata_paths):
    lines = []
    for metadata_path in metadata_paths:
        with open(metadata_path, 'r') as f:
            line = f.read().splitlines()
            lines.extend(line)
    lines = [l.split('|')[0] for l in lines]
    lines = [os.path.dirname(l) for l in lines]
    lines = list(set(lines))

    return {l: str(uuid.uuid4())[:8] for l in lines}

try:
    folder_dict = json.load(open(os.path.join(out_dir, "folder_dict.json"), "r"))
    print(f"Folder dict loaded from {os.path.join(out_dir, 'folder_dict.json')}")
except:
    folder_dict = prepare_folder_dict(txt_files)
    with open(os.path.join(out_dir, "folder_dict.json"), "w") as f:
        json.dump(folder_dict, f)
    print(f"Folder dict saved to {os.path.join(out_dir, 'folder_dict.json')}")

t = time()
lines = []
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        line = f.read().splitlines()
        lines.extend(line)
print(f"Time to read metadata: {time() - t}s")

def get_audio_duration(audio_path):
    try:
        audio = File(audio_path)
        return audio.info.length
    except:
        print(f"Error getting duration of {audio_path}")
        return None

def process_line(line):
    audio_path = line.split("|")[0]
    duration = get_audio_duration(audio_path)
    if duration is None:
        print(f"Error getting duration of {audio_path}")
        return None

    if duration < 0.8 or duration > 30:
        print(f"Duration of {audio_path} is {duration}, skipping")
        return None
    
    audio_folder = os.path.dirname(audio_path)
    audio_folder_id = folder_dict[audio_folder]
    audio_path = os.path.basename(audio_path)

    text = line.split("|")[1]
    if PROCESSING_JAPANESE:
        text = convert_char_to_charlist([text])[0]
    else:
        text = convert_char_to_pinyin([text], polyphone=True)[0]
    text = "|".join(text)
    if not os.path.exists(os.path.join(out_dir, "text", audio_folder_id)):
        os.makedirs(os.path.join(out_dir, "text", audio_folder_id), exist_ok=True)
    with open(os.path.join(out_dir, "text", audio_folder_id, f"{os.path.splitext(audio_path)[0]}.txt"), "w") as f:
        f.write(text)

    return f"{os.path.join(audio_folder_id, audio_path)}|{duration}"

with Pool(cpu_count()) as p:
    results = list(tqdm(p.imap_unordered(process_line, lines), total=len(lines)))
results = [r for r in tqdm(results) if r is not None]

with open(os.path.join(out_dir, "filelist.txt"), "w") as f:
    f.write("\n".join(results))

results = open(os.path.join(out_dir, "filelist.txt"), "r").read().splitlines()

# Create duration file
duration_json = {
    "duration": [float(r.split("|")[1]) for r in results]
}

with open(os.path.join(out_dir, "duration.json"), "w") as f:
    json.dump(duration_json, f)
