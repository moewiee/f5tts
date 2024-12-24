import os
import re
import torch
import torchaudio
import gradio as gr
import numpy as np
import tempfile
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_charlist,
    save_spectrogram,
)
from transformers import pipeline
import librosa
import click
import soundfile as sf
import Levenshtein
import string
from tqdm import tqdm
import json

import MeCab


# Specify the path to the IPAdic dictionary
dic_path = '/var/lib/mecab/dic/ipadic-utf8/'  # Replace this with the actual path you found

# Create a MeCab tagger object with the specified dictionary
tagger = MeCab.Tagger(f'-d {dic_path} -Ochasen')

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 64  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

def japanese_to_katakana(text):
    # Parse the text
    node = tagger.parseToNode(text)
    
    # Collect katakana outputs, keeping Latin characters as they are
    output = []
    
    while node:
        # Extract features and split into parts
        features = node.feature.split(',')
        
        if len(features) > 7:
            if features[1] != "記号":
                reading = features[7]
                if reading.isascii():  # Check if it's ASCII
                    output.append(node.surface)  # Append the original Latin characters
                else:
                    output.append(reading)  # Append Katakana reading
            else:
                output.append(node.surface)  # Include symbols directly
        else:
            output.append(node.surface)  # Fallback to the original text if no reading is found
        
        node = node.next

    return ''.join(output)

def load_model(ckpt_path, model_cls, model_cfg):
    vocab_char_map, vocab_size = get_tokenizer("japanese_dataset", "char")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model

# load models
F5TTS_ema_model = load_model(
   "/home/robert/E2-F5-TTS/ckpts/F5TTS_Base_834f17c7/model_last.pt", 
   DiT, 
   dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
)

def split_text_into_batches(text, max_chars=200, split_words=[]):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'
        
    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""
    
    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def infer_batch(ref_audio, ref_text, gen_text_batches, remove_silence):
    ema_model = F5TTS_ema_model

    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(gen_text_batches):
        # Prepare the text
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_charlist(text_list)

        # Calculate duration
        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu().float())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves
    final_wave = np.concatenate(generated_waves)

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path

def infer(ref_audio_orig, ref_text, gen_text, remove_silence, custom_split_words=''):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (30 - audio.shape[-1] / sr))
    gen_text_batches = split_text_into_batches(gen_text, max_chars=max_chars)
    
    return infer_batch((audio, sr), ref_text, gen_text_batches, remove_silence)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device,
)

# ref_audio = "rena-neutral_enhanced.wav"
# ref_text = "まず午前中には重要なクライアントとの会議が予定されています。そして午後にはプロジェクトの進捗報告がございます。"

ref_audio = "cut_20241223_151606.wav"
ref_text = "まず午前中には"

texts = open("data/500_test.txt").read().splitlines() + open("data/500_test_2.txt").read().splitlines()

def evaluate(ref_audio, ref_text, gen_text):
    audio, _ = infer(ref_audio, ref_text, gen_text, remove_silence=True)
    sf.write("output.wav", audio[1], audio[0])

    transcribed_text = pipe(
                "output.wav",
                chunk_length_s=30,
                batch_size=128,
                generate_kwargs={"task": "transcribe"},
                return_timestamps=False,
            )["text"].strip()

    # Calculate normalized Levenshtein distance after remove punctuation and Japanese punctuation
    gen_text = gen_text.translate(str.maketrans('', '', string.punctuation + '。，、；：？！'))
    transcribed_text = transcribed_text.translate(str.maketrans('', '', string.punctuation + '。，、；：？！'))

    levenshtein_distance = Levenshtein.distance(gen_text, transcribed_text)
    normalized_distance = levenshtein_distance / max(len(gen_text), len(transcribed_text))
    return normalized_distance, gen_text, transcribed_text, levenshtein_distance, max(len(gen_text), len(transcribed_text))

results = []
score = [0, 0]
tbar = tqdm(texts)
for text in tbar:
    distance, gen_text, transcribed_text, wrong_chars, total_chars = evaluate(ref_audio, ref_text, text)
    
    # Check if there are a-zA-Z0-9 in the transcribed text
    if not any(char.isascii() for char in transcribed_text):
        results.append({
            "gen_text": gen_text,
            "transcribed_text": transcribed_text,
            "distance": distance,
            "wrong_chars": wrong_chars,
            "total_chars": total_chars
        })
        score[0] += wrong_chars
        score[1] += total_chars
    else:
        print(f"Skipping {transcribed_text} because it contains alphanumeric characters")
    # Update score to the tqdm progress bar
    tbar.set_description(f"Score: {score[0] / score[1]}")

print(f"Score: {score[0] / score[1]}")
results = [f for f in results if f["distance"] != 0]
with open("results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 450k steps, short prompt, 0.0249 WER
# 500k steps, short prompt, 0.0256 WER