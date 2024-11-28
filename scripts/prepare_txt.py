# We use TXT since it take less RAM.

from time import time
import json
import os
from mutagen import File
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from model.utils import convert_char_to_pinyin

txt_files = [
    '/shared-data/tts/stable_filelist/emilia_english.txt',
    '/shared-data/tts/stable_filelist/english_s3_examples.txt'
]

folder_dict = json.load(open("folder_dict.json", "r"))

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
        return audio.info.length if audio else None
    except:
        print(f"Error getting duration of {audio_path}")
        return None

def process_line(line):
    audio_path = line.split("|")[0]
    duration = get_audio_duration(audio_path)
    if duration is None:
        return None

    if duration < 1 or duration > 30:
        return None
    
    audio_folder = os.path.dirname(audio_path)
    audio_folder_id = folder_dict[audio_folder]
    audio_path = os.path.basename(audio_path)

    text = line.split("|")[1]
    text = convert_char_to_pinyin([text], polyphone=True)[0]
    text = "|".join(text)
    if not os.path.exists(os.path.join("s3_emilia_text", audio_folder_id)):
        os.makedirs(os.path.join("s3_emilia_text", audio_folder_id), exist_ok=True)
    with open(os.path.join("s3_emilia_text", audio_folder_id, f"{os.path.splitext(audio_path)[0]}.txt"), "w") as f:
        f.write(text)

    return f"{os.path.join(audio_folder_id, audio_path)}|{duration}"

with Pool(cpu_count()) as p:
    results = list(tqdm(p.imap_unordered(process_line, lines), total=len(lines)))
results = [r for r in tqdm(results) if r is not None]
with open("s3_emilia_english.txt", "w") as f:
    f.write("\n".join(results))