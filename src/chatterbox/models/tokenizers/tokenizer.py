import logging

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    """Class for tokenizing English text using a vocabulary file.
    Initializes with a path to the vocabulary file and checks for special tokens SOT and EOT.
    Provides method to convert text into token IDs.
    """
    def __init__(self, vocab_file_path):
        """Initializes a tokenizer from a vocabulary file and checks for specific tokens.
        Args:
        vocab_file_path (str): Path to the vocabulary file.
        Checks if Start-Of-Token (SOT) and End-Of-Token (EOT) are present in the vocabulary.
        Encodes text into tokens.
        Args:
        txt (str): Input text to encode.
        verbose (bool, optional): If True, print additional information. Default is False.
        Returns:
        torch.IntTensor: Encoded text as a 1D tensor of integers.
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        """Checks if Start-Of-Turn and End-Of-Turn tokens are in the vocabulary.
        Args:
        None
        Returns:
        None
        ---
        Converts a string to token IDs.
        Args:
        text (str): The input text to convert.
        Returns:
        torch.Tensor: A tensor of token IDs.
        ---
        Encodes a string into token IDs using the Tokenizer.
        Args:
        txt (str): The text to encode.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        Returns:
        None
        """
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        """Converts text to tokens using a tokenizer.
        Args:
        text (str): The input text to be tokenized.
        Returns:
        torch.Tensor: A tensor containing the token IDs.
        """
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        """Decodes a sequence of tokens into a string.
        Args:
        seq (torch.Tensor): The input sequence to decode.
        Returns:
        str: The decoded string.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
