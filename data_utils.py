import numpy as np
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset

## Adapted from "Hemlata Tak, Jee-weon Jung - tak@eurecom.fr, jeeweon.jung@navercorp.com"

AUDIO_FILE_FIELD = 1
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


class Dataset_ASVspoof2019(Dataset):
    def __init__(self, config: dict, sample_name: str):
        # Read the data set protocol file.
        protocol_file = config[sample_name + "_protocol"]
        audio_files_folder = config[sample_name + "_audio_folder"]
        with open(protocol_file, "r", encoding="utf-8") as f:
            data_set_items = [x.strip().split() for x in f.readlines()]

        self.max_speech_length = config['max_speech_length']
        self.audio_files = [x[AUDIO_FILE_FIELD] for x in data_set_items]
        self.labels = [LABELS_MAP[x[LABEL_FIELD]] for x in data_set_items]
        self.audio_files_folder = audio_files_folder

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        signal, _ = sf.read(self.audio_files_folder + audio_file + ".flac")
        padded_signal = pad_random(signal, self.max_speech_length)
        tensor_signal = self.signal_to_tensor(padded_signal)
        y = self.labels[index]
        return tensor_signal, y
    
    def signal_to_tensor(self, signal: np.ndarray) -> Tensor:
        raise NotImplementedError()
