import torch

from abc import ABC, abstractmethod


def get_criterion(criterion_name: str):
    if criterion_name == "CTCLoss":
        return CTCLoss()
    elif criterion_name == "CrossEntropy":
        return CrossEntropy()

    return None


class Criterion(ABC):
    @abstractmethod
    def __call__(self, output, enc_pad_texts, text_lens):
        """
        :param output: output of model - [batch, seq_len, num_classes (alphabet_size)]
        :param enc_pad_texts: encoded target texts
        :param text_lens: length of encoded target texts without padding
        :return: double value equal of loss function output
        """
        pass

    @abstractmethod
    def to(self, device):
        """
        :param device: cuda or cpu
        :return: self
        """
        pass


class CTCLoss(Criterion):
    def __init__(self):
        self.loss_function = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def __call__(self, output, enc_pad_texts, text_lens):
        output_lengths = torch.full(
            size=(output.size(0),),
            fill_value=output.size(1),
            dtype=torch.long
        )
        return self.loss_function(output.permute(1, 0, 2), enc_pad_texts, output_lengths, text_lens)

    def to(self, device):
        self.loss_function.to(device)
        return self


class CrossEntropy(Criterion):
    def __init__(self):
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignore [GO] token = ignore index 0

    def __call__(self, output, enc_pad_texts, text_lens):
        return self.loss_function(output.view(-1, output.shape[-1]), enc_pad_texts.contiguous().view(-1))

    def to(self, device):
        self.loss_function.to(device)
        return self
