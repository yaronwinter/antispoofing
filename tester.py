import numpy as np
import torch
import time
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_utils
import aasist
import cnn

def compute_eer(model, test_set: DataLoader) -> tuple:
    t0 = time.time()
    active_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.eval()
    scores = []
    targets = []
    for signals, labels in tqdm(test_set):
        signals = signals.to(active_device)
        labels = labels.to(active_device)
        with torch.no_grad():
            curr_scores = model.predict(signals)
        curr_scores = curr_scores.detach().cpu().numpy()
        curr_scores = curr_scores[:,1] - curr_scores[:,0]
        curr_scores = curr_scores.clip(min=0)

        scores.append(curr_scores)
        targets.append(labels.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)
    fpr, tpr, _ = metrics.roc_curve(targets, scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fpr - fnr))
        
    return np.mean((fpr[eer_index], fnr[eer_index]))*100, (time.time() - t0)

def evaluate_sample(model, data_loader, eval_protocol: str, signals_folder: str, config: dict):
    model.eval()
    with open(eval_protocol, "r", encoding="utf-8") as f:
        data_set_items = [x.strip().split() for x in f.readlines()]

    with open(config['eval_out_file'], "w", encoding="utf-8") as f:
        f.write("file_name\tattack\tlabel\tspoof_score\tbonafide_score\n")
        for item in tqdm(data_set_items):
            audio_file = item[data_utils.AUDIO_FILE_FIELD]
            label = item[data_utils.LABEL_FIELD]
            attack = item[data_utils.ATTACK_TYPE_FIELD]
            signal = data_loader.read_signal(signals_folder + audio_file + ".flac")
            with torch.no_grad():
                scores = model.predict(signal)

            f.write(audio_file + "\t" + attack + "\t" + label + "\t{:.3f}".format(scores[0,0].item()) + "\t{:.3f}".format(scores[0,1].item()) + "\n")
            f.flush()

def load_model(model_type: str, model_path: str, config: dict) -> tuple:
    if model_type == "aasist":
        model = aasist.model.Model(config)
        data_loader = aasist.data_loader.AAsistDataLoader(config, None)
    elif model_type == "cnn":
        data_loader = cnn.data_loader.CNNDataLoader(config, None)
        model = cnn.cnn.CNN(config, 2)
    else:
        raise NotImplementedError("Unknown model name was given: " + model_type)
    model.load_state_dict(torch.load(model_path))
    return model.eval(), data_loader


def evaluate_signal(model_type: str, model_path: str, label: str, attack: str, config: dict) -> str:
    print("load the model")
    model, data_loader = load_model(model_type, model_path, config)
    
    print("read the signal")
    signal = data_loader.get_eval_sample(label, attack, config)

    print("compute signal")
    with torch.no_grad():
        scores = model.predict(signal)
    results = "\nspoof score={:.3f}".format(scores[0,0].item())
    results += "\nbonafide score={:.3f}".format(scores[0,1].item())
    return results
