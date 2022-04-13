from utils.config import Config

from models import vanilla_crnn, vitstr


def get_model(model_name: str, number_class_symbols: int, config: Config = None, **kwargs):
    if model_name == "CRNN":
        return vanilla_crnn.CRNN(number_class_symbols, **kwargs)
    elif model_name == "ViTSTR":
        return vitstr.create_vitstr(number_class_symbols=number_class_symbols,
                                    model=config.get('transformer_model'),
                                    output_length=config.get('output_length'),
                                    in_chans=3, **kwargs)

    return None
