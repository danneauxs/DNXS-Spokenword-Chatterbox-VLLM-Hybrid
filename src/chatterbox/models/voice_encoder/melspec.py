from functools import lru_cache

from scipy import signal
import numpy as np
import librosa


@lru_cache()
def mel_basis(hp):
    """Calculates the Mel-frequency cepstral coefficients (MFCCs) of an audio signal.
    Args:
    - hp: A dictionary containing hyperparameters including sample_rate, n_fft, num_mels, fmin, and fmax.
    Returns: An array of MFCCs with shape (nmel, nfreq).
    Applies preemphasis to an audio signal to filter out high-frequency noise.
    Args:
    - wav: The input audio signal as a numpy array.
    - hp: A dictionary containing the preemphasis coefficient.
    Returns: The preemphasized audio signal clipped between -1 and 1.
    """
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


def preemphasis(wav, hp):
    """Applies preemphasis to a waveform.
    Args:
    wav: Input audio waveform as a numpy array.
    hp: Hyperparameters containing the preemphasis coefficient.
    Returns:
    Preemphasized audio waveform as a numpy array.
    """
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    """Computes the mel-spectrogram of a given WAV audio signal.
    Args:
    wav (numpy.ndarray): The input audio waveform.
    hp (dict): A dictionary containing hyperparameters.
    pad (bool, optional): Whether to pad the input audio waveform. Defaults to True.
    Returns:
    numpy.ndarray: The computed mel-spectrogram.
    """
    # Run through pre-emphasis
    if hp.preemphasis > 0:
        wav = preemphasis(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    # Do the stft
    spec_complex = _stft(wav, hp, pad=pad)

    # Get the magnitudes
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    # Get the mel and convert magnitudes->db
    mel = np.dot(mel_basis(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db(mel, hp)

    # Normalise the mel from db to 0,1
    if hp.normalized_mels:
        mel = _normalize(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size   # Sanity check
    return mel   # (M, T)


def _stft(y, hp, pad=True):
    """Calculates the Short-Time Fourier Transform (STFT) of an audio signal.
    Args:
    y: The input audio time series.
    hp: A dictionary containing hyperparameters such as n_fft, hop_size, and win_size.
    pad: If True, pads the signal to make its length a multiple of the frame length.
    Returns:
    The STFT result of the input audio signal.
    """
    # NOTE: after 0.8, pad mode defaults to constant, setting this to reflect for
    #   historical consistency and streaming-version consistency
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def _amp_to_db(x, hp):
    """Converts amplitude to decibels.
    Args:
    x (float): Amplitude value.
    hp (object): Hyperparameters object containing stft_magnitude_min attribute.
    Returns:
    float: Decibel value.
    Converts decibels to amplitude.
    Args:
    x (float): Decibel value.
    Returns:
    float: Amplitude value.
    Normalizes the input signal s.
    Args:
    s (float): Input signal.
    hp (object): Hyperparameters object containing stft_magnitude_min attribute.
    headroom_db (float, optional): Headroom in decibels. Defaults to 15.
    Returns:
    float: Normalized signal.
    """
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    """Converts a decibel value to an amplitude value.
    Args:
    x (float): The decibel value to convert.
    Returns:
    float: The corresponding amplitude value.
    ---
    Normalizes an audio signal.
    Args:
    s (numpy.ndarray): The input audio signal.
    hp (HyperParameters): Hyperparameters containing the minimum STFT magnitude.
    headroom_db (int, optional): Headroom in decibels. Defaults to 15.
    Returns:
    numpy.ndarray: The normalized audio signal.
    """
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    """Normalizes an audio signal based on its minimum magnitude and a headroom level.
    Args:
    s (np.ndarray): The input audio signal.
    hp (Hparams): Hyperparameters containing the STFT magnitude minimum.
    headroom_db (float, optional): Headroom in decibels above the minimum level. Defaults to 15.
    Returns:
    np.ndarray: The normalized audio signal.
    """
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
