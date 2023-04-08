import numpy as np
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset

## Adapted from "Hemlata Tak, Jee-weon Jung - tak@eurecom.fr, jeeweon.jung@navercorp.com"

AUDIO_FILE_FIELD = 1
ATTACK_TYPE_FIELD = 3
LABEL_FIELD = 4
LABELS_MAP = {"bonafide":1, "spoof":0}

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def get_eval_sample(label: str, attack_type: str, config: dict) -> Tensor:
        protocol_file = config["eval_protocol"]
        audio_files_folder = config["eval_audio_folder"]
        with open(protocol_file, "r", encoding="utf-8") as f:
            data_set_items = [x.strip().split() for x in f.readlines()]

        if label is not None:
            data_set_items = [x for x in data_set_items if x[LABEL_FIELD] == label]

        if attack_type is not None:
            data_set_items = [x for x in data_set_items if x[ATTACK_TYPE_FIELD] == attack_type]

        np.random.shuffle(data_set_items)
        audio_file = data_set_items[0][AUDIO_FILE_FIELD]
        return read_signal(audio_files_folder + audio_file + ".flac", config['max_speech_elngth'])

def read_signal(signal_file: str, max_speech_length) -> Tensor:
    signal, _ = sf.read(signal_file)
    padded_signal = pad_random(signal, max_speech_length)
    tensor_signal = Tensor(padded_signal)
    return tensor_signal.unsqueeze(dim=0)


class Dataset_ASVspoof2019(Dataset):
    def __init__(self, config: dict, sample_name: str):
        # Read the data set protocol file.
        protocol_file = config[sample_name + "_protocol"]
        audio_files_folder = config[sample_name + "_audio_folder"]
        with open(protocol_file, "r", encoding="utf-8") as f:
            data_set_items = [x.strip().split() for x in f.readlines()]

        self.max_speech_length = config['max_speech_length']
        self.audio_files = [x[AUDIO_FILE_FIELD] for x in data_set_items]
        self.labels = [x[LABEL_FIELD] for x in data_set_items]
        self.audio_files_folder = audio_files_folder

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        label = self.labels[index]
        signal, _ = sf.read(self.audio_files_folder + audio_file + ".flac")
        padded_signal = pad_random(signal, self.max_speech_length)
        tensor_signal = self.signal_to_tensor(padded_signal)
        y = LABELS_MAP[label]
        return tensor_signal, y
    
    def signal_to_tensor(self, signal: np.ndarray) -> Tensor:
        raise NotImplementedError()
