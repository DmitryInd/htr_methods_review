from abc import ABC, abstractmethod

from utils.config import Config

OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'
GO = '<GO>'
SPACE = '<s>'


def get_tokenizer(tokenizer_name: str, config: Config):
    if tokenizer_name == "CTC":
        return CTCTokenizer(config.get('alphabet'))
    elif tokenizer_name == "Base":
        return BaseTokenizer(config.get('alphabet'), config.get('output_length'))

    return None


class Tokenizer(ABC):
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    @abstractmethod
    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        pass

    @abstractmethod
    def get_num_chars(self):
        pass

    @abstractmethod
    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        pass


class CTCTokenizer(Tokenizer):
    def __init__(self, alphabet):
        self.char_map = self._get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    @staticmethod
    def _get_char_map(alphabet):
        """Make from string alphabet character2int dict.
        Add BLANK char for CTC loss and OOV char for out of vocabulary symbols."""
        char_map = {value: idx + 2 for (idx, value) in enumerate(alphabet)}
        char_map[CTC_BLANK] = 0
        char_map[OOV_TOKEN] = 1
        return char_map

    def encode(self, word_list):
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                    char_enc != self.char_map[OOV_TOKEN]
                    and char_enc != self.char_map[CTC_BLANK]
                    # idx > 0 to avoid selecting [-1] item
                    and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class BaseTokenizer(Tokenizer):
    def __init__(self, alphabet, output_length):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.

        self.list_token = [GO, OOV_TOKEN, SPACE]
        self.character = self.list_token + list(alphabet)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = output_length

    def encode(self, text):
        # length = [len(s) + 2 for s in text]: + 2 for [GO] and [s] at end of sentence
        batch_text = [[self.dict[GO]] * (self.batch_max_length + 2) for _ in text]
        for i, t in enumerate(text):
            assert len(t) <= self.batch_max_length, f"This sequence is too long:\n{t}"
            txt = [GO] + list(t) + [SPACE]
            txt = [self.dict[char] if char in self.character else self.dict[OOV_TOKEN] for char in txt]
            batch_text[i][:len(txt)] = txt  # batch_text[:, 0] = [GO] token
        return batch_text

    def get_num_chars(self):
        return len(self.character)

    def decode(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for enc_str in text_index:
            text = ''
            for char_enc in enc_str:
                # skip space, out of vocabulary and go token, stop if space symbol
                if char_enc != self.dict[GO] and char_enc != self.dict[OOV_TOKEN] and char_enc != self.dict[SPACE]:
                    text += self.character[char_enc]
                elif char_enc == self.dict[SPACE]:
                    break
            texts.append(text)
        return texts
