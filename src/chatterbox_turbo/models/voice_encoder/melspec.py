from functools import lru_cache

from scipy import signal
import numpy as np
import librosa


@lru_cache()
def mel_basis(hp):
    """Converts a waveform to mel-spectrogram.
    Args:
    hp (dict): Hyperparameters containing sample rate, FFT size, number of Mel bands, minimum frequency, and maximum frequency.
    Returns:
    np.ndarray: Mel-spectrogram of shape (nmel, nfreq).
    """
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax)  # -> (nmel, nfreq)


def preemphasis(wav, hp):
    """Applies pre-emphasis to an audio waveform.
    Args:
    wav (np.ndarray): Input audio waveform.
    hp (hparams.HParams): Hyperparameters object containing preemphasis settings.
    Returns:
    np.ndarray: Pre-emphasized audio waveform.
    """
    assert hp.preemphasis != 0
    wav = signal.lfilter([1, -hp.preemphasis], [1], wav)
    wav = np.clip(wav, -1, 1)
    return wav


def melspectrogram(wav, hp, pad=True):
    """Computes the mel-spectrogram of a given audio waveform.
    Args:
    wav (np.ndarray): Audio waveform.
    hp (dict): Hyperparameters containing preprocessing settings.
    pad (bool, optional): Whether to pad the waveform before STFT. Defaults to True.
    Returns:
    np.ndarray: Mel-spectrogram.
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
    """Compute the Short-Time Fourier Transform (STFT) of an audio signal.
    Args:
    y (np.ndarray): Audio time series.
    hp (object): Hyperparameters object containing STFT settings.
    Returns:
    np.ndarray: STFT matrix.
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
    x: Input amplitude value(s).
    hp: Hyperparameters containing magnitude minimum for STFT.
    Returns: Decibel value(s).
    Converts decibels back to amplitude.
    Args:
    x: Input decibel value(s).
    Returns: Amplitude value(s).
    Normalizes the input signal by converting its amplitude to a scale between 0 and 1.
    Args:
    s: Input signal amplitude.
    hp: Hyperparameters containing magnitude minimum for STFT.
    headroom_db: Headroom in decibels for normalization.
    Returns: Normalized signal.
    """
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _db_to_amp(x):
    """Converts decibel values to amplitude.
    Args:
    x (float): Decibel value.
    Returns:
    float: Amplitude value.
    Normalizes a signal based on its minimum level and headroom in decibels.
    Args:
    s (np.ndarray): Input signal.
    hp (object): Hyperparameters object containing stft_magnitude_min.
    headroom_db (int, optional): Headroom in decibels. Defaults to 15.
    Returns:
    np.ndarray: Normalized signal.
    """
    return np.power(10.0, x * 0.05)


def _normalize(s, hp, headroom_db=15):
    """Normalizes a signal by scaling it relative to a minimum level and headroom.
    Args:
    s (np.ndarray): The input signal.
    hp (dict): Hyperparameters including 'stft_magnitude_min'.
    headroom_db (float, optional): Headroom in decibels. Default is 15.
    Returns:
    np.ndarray: Normalized signal.
    """
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    s = (s - min_level_db) / (-min_level_db + headroom_db)
    return s
