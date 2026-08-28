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
    """A tokenizer for English text using a vocabulary file.
    Initializes with a path to a vocabulary file and checks for start-of-text (SOT) and end-of-text (EOT) tokens in the vocabulary.
    Converts input text to token IDs, wraps them in a tensor, and adds an extra dimension.
    """
    def __init__(self, vocab_file_path):
        """Initializes a tokenizer from a vocabulary file and checks for special tokens SOT and EOT.
        Args:
        - vocab_file_path (str): Path to the vocabulary file.
        Returns:
        None
        text_to_tokens converts a text string to token IDs.
        Args:
        - text (str): Input text to tokenize.
        Returns:
        torch.Tensor: Tensor of token IDs.
        encode encodes a given text into tokens, optionally providing verbose output.
        Args:
        - txt (str): The input text to encode.
        - verbose (bool): Whether to print detailed encoding information.
        Returns:
        list: List of token IDs.
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        """Checks if Start of Text and End of Text tokens are in the vocabulary.
        Args:
        None
        Returns:
        None
        Converts a string to token IDs using tokenizer.
        Args:
        text (str): Input text to convert
        Returns:
        torch.IntTensor: Tensor of token IDs
        Cleans and encodes input text.
        Args:
        txt (str): Text to encode
        verbose (bool, optional): Whether to print verbose output. Defaults to False
        Returns:
        str: Encoded text
        """
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        """Converts a string of text into token IDs.
        Args:
        text: A string to be converted.
        Returns:
        A PyTorch tensor of shape (1, n) containing the token IDs.
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
        """Decodes a sequence into a string.
        Args:
        seq (torch.Tensor or list of int): The sequence to decode.
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
