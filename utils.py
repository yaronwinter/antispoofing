import os
import shutil
from tqdm import tqdm
import data_utils
import numpy as np

def generate_sub_sample(config: dict):
    print("generate sub sample - start")
    sub_sample_folder = config['sub_sample_folder']

    if os.path.exists(sub_sample_folder):
        shutil.rmtree(sub_sample_folder)
    os.makedirs(sub_sample_folder)
    protocols = [config['train_protocol'], config['dev_protocol'], config['eval_protocol']]
    audio_folders = [config['train_audio_folder'], config['dev_audio_folder'], config['eval_audio_folder']]
    fractions = [config['train_fraction'], config['dev_fraction'], config['eval_fraction']]
    sample_names = ['train', 'dev', 'eval']
    for protocol, audio_folder, fraction, name in zip(protocols, audio_folders, fractions, sample_names):
        handle_sample(protocol, audio_folder, fraction, name, sub_sample_folder)
    print("generate sub sample - end")

def handle_sample(protocol: str, audio_folder: str, fraction: float, sample_name: str, sub_sample_folder: str):
    print("handle sample: " + sample_name)
    with open(protocol, "r", encoding="utf-8") as f:
        items = [x.strip() for x in f.readlines()]

    sub_sample = [x for x in items if np.random.random() < fraction]
    new_audio_folder = sub_sample_folder + sample_name + "/"
    os.makedirs(sub_sample_folder + sample_name)
    with open(sub_sample_folder + sample_name + "_protocol.txt", "w", encoding='utf-8') as f:
        for sample in tqdm(sub_sample):
            f.write(sample + "\n")
            f.flush()
            audio_file = sample.split()[data_utils.AUDIO_FILE_FIELD]
            shutil.copy2(audio_folder + audio_file + ".flac", new_audio_folder)
