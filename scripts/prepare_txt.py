# We use TXT since it take less RAM.

from time import time
import json
import os
from mutagen import File
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from model.utils import convert_char_to_pinyin, convert_char_to_charlist
import uuid

import MeCab
from furigana import make_furigana

PROCESSING_JAPANESE = True
# Specify the path to the IPAdic dictionary
dic_path = '/var/lib/mecab/dic/ipadic-utf8/'  # Replace this with the actual path you found

# Create a MeCab tagger object with the specified dictionary
tagger = MeCab.Tagger(f'-d {dic_path} -Ochasen')

# List of input text files containing metadata
txt_files = [
    '/mnt/weka/tts/jchat_f5/content.txt',
    '/mnt/md0/tts/data/jaudiobook.txt',
    '/mnt/md0/tts/data/jemilia/content.txt',
    '/mnt/md0/tts/data/jreazon.txt',
]

# Output directory for processed files
out_dir = 'data/japanese_dataset_char'

# Function to create a mapping of folders to unique IDs
def prepare_folder_dict(metadata_paths):
    # Initialize empty list for all lines
    lines = []
    # Read each metadata file
    for metadata_path in metadata_paths:
        with open(metadata_path, 'r') as f:
            line = f.read().splitlines()
            lines.extend(line)
    # Extract audio paths from lines
    lines = [l.split('|')[0] for l in lines]
    # Get unique folder paths
    lines = [os.path.dirname(l) for l in lines]
    lines = list(set(lines))
    
    # Create dictionary mapping folders to random 8-char UUIDs
    return {l: str(uuid.uuid4())[:8] for l in lines}

# Try to load existing folder dictionary, create new one if not found
try:
    folder_dict = json.load(open(os.path.join(out_dir, "folder_dict.json"), "r"))
    print(f"Folder dict loaded from {os.path.join(out_dir, 'folder_dict.json')}")
except:
    folder_dict = prepare_folder_dict(txt_files)
    with open(os.path.join(out_dir, "folder_dict.json"), "w") as f:
        json.dump(folder_dict, f)
    print(f"Folder dict saved to {os.path.join(out_dir, 'folder_dict.json')}")

# Time the metadata reading process
t = time()
lines = []
# Read all lines from input files
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        line = f.read().splitlines()
        lines.extend(line)
print(f"Time to read metadata: {time() - t}s")

# Function to get audio duration using mutagen
def get_audio_duration(audio_path):
    try:
        audio = File(audio_path)
        return audio.info.length
    except:
        print(f"Error getting duration of {audio_path}")
        return None

# Function to process each line of metadata
def process_line(line):
    # Extract audio path
    audio_path = line.split("|")[0]
    # Get duration from metadata or audio file
    if line.split("|").__len__() < 3:
        duration = get_audio_duration(audio_path)
    else:
        duration = float(line.split("|")[2])
    
    # Skip if duration couldn't be determined
    if duration is None:
        print(f"Error getting duration of {audio_path}")
        return None

    # Skip if duration outside valid range
    if duration < 0.8 or duration > 30:
        print(f"Duration of {audio_path} is {duration}, skipping")
        return None
    
    # Get folder ID and process paths
    audio_folder = os.path.dirname(audio_path)
    audio_folder_id = folder_dict[audio_folder]
    audio_path = os.path.basename(audio_path)

    # Process text based on language
    text = line.split("|")[1]
    if PROCESSING_JAPANESE:
        text = make_furigana(text, tagger=tagger)
        text = convert_char_to_charlist([text])[0]
    else:
        text = convert_char_to_pinyin([text], polyphone=True)[0]
    text = "|".join(text)
    
    # Create output directory and save processed text
    if not os.path.exists(os.path.join(out_dir, "text", audio_folder_id)):
        os.makedirs(os.path.join(out_dir, "text", audio_folder_id), exist_ok=True)
    with open(os.path.join(out_dir, "text", audio_folder_id, f"{os.path.splitext(audio_path)[0]}.txt"), "w") as f:
        f.write(text)

    # Return processed audio path and duration
    return f"{os.path.join(audio_folder_id, audio_path)}|{duration}"

# Process all lines in parallel using multiprocessing
with Pool(cpu_count()) as p:
    results = list(tqdm(p.imap_unordered(process_line, lines), total=len(lines)))
# Filter out None results
results = [r for r in tqdm(results) if r is not None]

# Save processed filelist
with open(os.path.join(out_dir, "filelist.txt"), "w") as f:
    f.write("\n".join(results))

# Read back the filelist
results = open(os.path.join(out_dir, "filelist.txt"), "r").read().splitlines()

# Create duration file with list of all durations
duration_json = {
    "duration": [float(r.split("|")[1]) for r in results]
}

# Save duration file
with open(os.path.join(out_dir, "duration.json"), "w") as f:
    json.dump(duration_json, f)
