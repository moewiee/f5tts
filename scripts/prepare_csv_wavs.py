import sys
import os
from pathlib import Path
import json
import shutil
import json
import argparse
import csv
import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter
from model.utils import convert_char_to_pinyin
from multiprocessing import Pool, cpu_count
from mutagen import File

PRETRAINED_VOCAB_PATH = Path(__file__).parent.parent / "data/Emilia_ZH_EN_pinyin/vocab.txt"
FOLDER_DICT_PATH = Path(__file__).parent.parent / "folder_dict.json"
FOLDER_DICT = json.load(open(FOLDER_DICT_PATH, "r"))

def is_csv_wavs_format(input_dataset_dir):
    fpath = Path(input_dataset_dir)
    metadata = fpath / "metadata.csv"
    wavs = fpath / "wavs"
    return metadata.exists() and metadata.is_file() and wavs.exists() and wavs.is_dir()

def get_audio_duration(audio_path):
    try:
        audio = File(audio_path)
        return audio.info.length if audio else None
    except:
        audio, sample_rate = torchaudio.load(audio_path)
        num_channels = audio.shape[0]
        return audio.shape[1] / (sample_rate * num_channels)

def process_line(line):
    audio_path, text = line.strip().split('|')
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
    except:
        return None
    text = convert_char_to_pinyin([text], polyphone=True)[0]
    # Get text path by changing the extension of audio path
    text_path = os.path.splitext(audio_path)[0] + ".txt"
    with open(text_path, "w") as f:
        f.write("".join(text))
    audio_folder = os.path.dirname(audio_path)
    audio_folder_id = FOLDER_DICT[audio_folder]
    audio_path = audio_path.replace(audio_folder, audio_folder_id)
    return {"audio_path": audio_path, "duration": audio_duration}

def prepare_csv_wavs_dir(metadata_paths):
    sub_result, durations = [], []
    vocab_set = set()
    lines = []
    for metadata_path in metadata_paths:
        with open(metadata_path, 'r') as f:
            line = f.read().splitlines()
            lines.extend(line)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_line, lines), total=len(lines), desc="Processing lines..."))

    for result in results:
        if result is not None:
            sub_result.append(result)
            durations.append(result["duration"])
            vocab_set.update(list(result["text"]))

    return sub_result, durations, vocab_set

def save_prepped_dataset(out_dir, result, duration_list, text_vocab_set, is_finetune):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=1) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    voca_out_path = out_dir / "vocab.txt"
    with open(voca_out_path.as_posix(), "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    if is_finetune:
        file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        shutil.copy2(file_vocab_finetune, voca_out_path)

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")

def prepare_and_save_set(inp_dir, out_dir, is_finetune: bool = True):
    if is_finetune:
        assert PRETRAINED_VOCAB_PATH.exists(), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    sub_result, durations, vocab_set = prepare_csv_wavs_dir(inp_dir)
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune)

def cli():
    medata_path = ['/shared-data/tts/stable_filelist/emilia_english.txt',
                    '/shared-data/tts/stable_filelist/english_s3_examples.txt'
    ]
    out_dir = '/mnt/md0/tts/tuna/E2-F5-TTS/data/emilia_s3_english_pinyin_reduced'
    prepare_and_save_set(medata_path, out_dir, is_finetune=True)

if __name__ == "__main__":
    cli()