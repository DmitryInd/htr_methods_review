import os
import torch


class FilesLimitControl:
    """Delete files from the disk if there are more files than the set limit.
    Args:
        max_weights_to_save (int, optional): The number of files that will be
            stored on the disk at the same time. Default is 3.
    """
    def __init__(self, max_weights_to_save=2):
        self.saved_weights_paths = []
        self.max_weights_to_save = max_weights_to_save

    def __call__(self, save_path):
        self.saved_weights_paths.append(save_path)
        if len(self.saved_weights_paths) > self.max_weights_to_save:
            old_weights_path = self.saved_weights_paths.pop(0)
            if os.path.exists(old_weights_path):
                os.remove(old_weights_path)
                print(f"Weights removed '{old_weights_path}'")


def load_pretrain_model(weights_path, model):
    """Load the entire pretrain model or as many layers as possible.
    """
    old_dict = torch.load(weights_path)
    new_dict = model.state_dict()
    for key, weights in new_dict.items():
        if key in old_dict:
            if new_dict[key].shape == old_dict[key].shape:
                new_dict[key] = old_dict[key]
            else:
                print('Weights {} were not loaded'.format(key))
        else:
            print('Weights {} were not loaded'.format(key))
    return new_dict
