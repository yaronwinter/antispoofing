import data_utils
from torch import Tensor
import numpy as np
import torchaudio

class CNNDataLoader(data_utils.Dataset_ASVspoof2019):
    def __init__(self, config: dict, sample_name: str):
        super().__init__(config, sample_name)

        self.speckwargs={"n_fft": config['fft_frame_size'], "hop_length": config['hop_length'], "center": False}
        self.transform = torchaudio.transforms.LFCC(sample_rate=config['sample_rate'], n_lfcc=config['lfcc_size'], speckwargs=self.speckwargs)

    def signal_to_tensor(self, signal: np.ndarray) -> Tensor:
        return self.transform(Tensor(signal))
