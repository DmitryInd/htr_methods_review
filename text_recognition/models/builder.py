import torch

from models import (vanilla_crnn, vitstr)


def get_model(model_name: str, **kwargs):
    if model_name == "CRNN":
        return vanilla_crnn.CRNN(**kwargs)
    elif model_name == "ViTSTR":
        return vitstr.create_vitstr(num_tokens=kwargs.get('num_class', 40),
                                    model=kwargs.get('TransformerModel', "vitstr_tiny_patch16_224"))
    return None


def get_criterion(criterion_name: str, **kwargs):
    if criterion_name == "CTCLoss":
        return torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    elif criterion_name == "CrossEntropy":
        return torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0
    return None
