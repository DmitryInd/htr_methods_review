import torch

from models import (vanilla_crnn)


def get_model(model_name: str, **kwargs):
    if model_name == "CRNN":
        return vanilla_crnn.CRNN(**kwargs)
    return None


def get_criterion(criterion_name: str, **kwargs):
    if criterion_name == "CTCLoss":
        return torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    return None
