import data_utils
from torch import Tensor
import numpy as np

class AAsistDataLoader(data_utils.Dataset_ASVspoof2019):
    def __init__(self, config: dict, sample_name: str):
        super().__init__(config, sample_name)

    def signal_to_tensor(self, signal: np.ndarray) -> Tensor:
        return Tensor(signal)
