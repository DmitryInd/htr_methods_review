import torch
import numpy as np

from utils.transforms import InferenceTransform
from utils.tokenizer import Tokenizer
from utils.config import Config
from utils.tokenizer import get_tokenizer
from models.builder import get_model


def get_text_from_probs(output, tokenizer):
    pred = torch.argmax(output.detach().cpu(), -1).numpy()
    return tokenizer.decode(pred)


def predict(images, model, tokenizer, device):
    """Make model prediction.

    Args:
        images (torch.Tensor): Batch with tensor images.
        model (utils.src.models.CRNN): OCR model.
        tokenizer (ocr.tokenizer.Tokenizer): Tokenizer class.
        device (torch.device): Torch device.
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    text_preds = get_text_from_probs(output, tokenizer)
    return text_preds


class OcrPredictor:
    def __init__(self, model_path, config_path, device='cuda'):
        config = Config(config_path)
        self.tokenizer = get_tokenizer(config.get('tokenizer'), config)
        self.device = torch.device(device)
        # load model
        self.model = get_model(config.get("model"), number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config.get_image('height'),
            width=config.get_image('width'),
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(f"Input must contain np.ndarray, "
                            f"tuple or list, found {type(images)}.")

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred
