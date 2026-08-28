from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """Class representing configuration settings for T3 models, including tokenization and model architecture details."""
    def __init__(self, text_tokens_dict_size=704):
        """Initializes tokenization and configuration settings for text and speech processing.
        Args:
        text_tokens_dict_size (int): Size of the dictionary for text tokens. Defaults to 704.
        Returns: None
        """
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

    @property
    def n_channels(self):
        """Returns the number of hidden channels in the current LLaMA configuration.
        Args:
        self (object): The instance of the class.
        Returns:
        int: The number of hidden channels.
        """
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        """Checks if the text tokens dictionary size matches that of a multilingual TTS model.
        Args:
        self: The instance of the class.
        Returns:
        bool: True if the text tokens dictionary size is 2454, False otherwise.
        Class methods:
        - english_only(cls): Returns configuration for an English-only TTS model.
        - multilingual(cls): Returns configuration for a multilingual TTS model.
        """
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        """Create configuration for English-only TTS model."""
        return cls(text_tokens_dict_size=704)
    
    @classmethod 
    def multilingual(cls):
        """Create configuration for multilingual TTS model."""
        return cls(text_tokens_dict_size=2454)
