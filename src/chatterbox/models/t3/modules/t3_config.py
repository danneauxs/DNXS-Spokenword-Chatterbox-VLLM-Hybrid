from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """Stores configuration parameters for T3 models, including token and dictionary sizes, as well as model-specific settings."""
    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    # For T3CondEnc
    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    @property
    def n_channels(self):
        """Returns the number of channels (hidden size) for the current LLaMA configuration."""
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
