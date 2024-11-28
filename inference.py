from api import F5TTS
from tqdm import tqdm
import os



with open('test.txt','r') as f:
    texts = f.read().splitlines()

prompts = {
    "angry": {
        'filepath': "/mnt/md0/tts/tuna/english_prompt/Emotion Speech Dataset/0019/Angry/0019_000363.wav",
        'text': "I have bunburyed all over goat on two separate occasions."
    },
    "happy": {
        'filepath': "/mnt/md0/tts/tuna/english_prompt/Emotion Speech Dataset/0019/Happy/0019_000729.wav",
        'text': "Rat came and replied on the leaves."
    },
    "neutral": {
        'filepath': "/mnt/md0/tts/tuna/english_prompt/Emotion Speech Dataset/0019/Neutral/0019_000033.wav",
        'text': "This used to be Jerry's occupation."
    },
    "sad":{
        'filepath': "/mnt/md0/tts/tuna/english_prompt/Emotion Speech Dataset/0019/Sad/0019_001095.wav",
        'text': "The nastiest things they saw were the cobwebs."
    },
    "surprise": {
        'filepath': "/mnt/md0/tts/tuna/english_prompt/Emotion Speech Dataset/0019/Surprise/0019_001433.wav",
        'text': "This used to be Jerry's occupation."
    },
}

f5tts = F5TTS()

for prompt in tqdm(prompts):
    prompt_path = prompts[prompt]['filepath']
    ref_text = prompts[prompt]['text']
    output_folder = f'/mnt/md0/tts/tuna/output_f5tts/{prompt}'
    os.makedirs(output_folder, exist_ok = True) 
    for i,text in tqdm(enumerate(texts)):

        
        output_file = os.path.join(output_folder,f'{i}.wav')

        wav, sr, spect = f5tts.infer(
            ref_file=prompt_path,
            ref_text=ref_text,
            gen_text=text,
            file_wave=output_file,
            file_spect="out.png",
            seed=-1,  # random seed = -1
        )