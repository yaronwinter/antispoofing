import numpy as np
import torch
import time
from sklearn import metrics
from torch.utils.data import DataLoader

def compute_eer(model: torch.nn.Module, test_set: DataLoader) -> tuple:
    t0 = time.time()
    active_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.eval()
    scores = []
    targets = []
    for signals, labels in test_set:
        with torch.no_grad():
            signals = signals.to(active_device)
            labels = labels.to(active_device)
            _, logits = model(signals)
        curr_scores = torch.softmax(logits, dim=1).detach().cpu().numpy()
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
