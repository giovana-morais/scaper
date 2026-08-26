# CREATED: 4/23/17 15:37 by Justin Salamon <justin.salamon@nyu.edu>

from pedalboard import Pedalboard, Limiter
import numpy as np
import pyloudnorm
import soundfile
from .scaper_exceptions import ScaperError


def get_integrated_lufs(audio_array, samplerate, min_duration=0.5,
                        filter_class='K-weighting', block_size=0.400):
    """
    Returns the integrated LUFS for a numpy array containing
    audio samples.

    For files shorter than 400 ms pyloudnorm throws an error. To avoid this, 
    files shorter than min_duration (by default 500 ms) are self-concatenated 
    until min_duration is reached and the LUFS value is computed for the 
    concatenated file.

    Parameters
    ----------
    audio_array : np.ndarray
        numpy array containing samples or path to audio file for computing LUFS
    samplerate : int
        Sample rate of audio, for computing duration
    min_duration : float
        Minimum required duration for computing LUFS value. Files shorter than
        this are self-concatenated until their duration reaches this value
        for the purpose of computing the integrated LUFS. Caution: if you set
        min_duration < 0.4, a constant LUFS value of -70.0 will be returned for
        all files shorter than 400 ms.
    filter_class : str
        Class of weighting filter used.
        - 'K-weighting' (default)
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
    block_size : float 
        Gating block size in seconds. Defaults to 0.400.
    
    Returns
    -------
    loudness
        Loudness in terms of LUFS 
    """
    duration = audio_array.shape[0] / float(samplerate)
    if duration < min_duration:
        ntiles = int(np.ceil(min_duration / duration))
        audio_array = np.tile(audio_array, (ntiles, 1))
    meter = pyloudnorm.Meter(
        samplerate, filter_class=filter_class, block_size=block_size
    )
    loudness = meter.integrated_loudness(audio_array)
    # silent audio gives -inf, so need to put a lower bound.
    loudness = max(loudness, -70) 
    return loudness


def match_sample_length(audio_path, duration_in_samples):
    '''
    Takes a path to an audio file and a duration defined in samples. The audio
    is loaded from the specifid audio_path and padded or trimmed such that it
    matches the duration_in_samples. The modified audio is then saved back to
    audio_path. This ensures that the durations match exactly. If the audio
    needed to be padded, it is padded with zeros to the end of the audio file.
    If the audio needs to be trimmed, the function will trim samples from the end of 
    the audio file. The sample rate of the saved audio is the same as the sample 
    rate of the input file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file that will be modified.
    duration_in_samples : int
        Duration that the audio will be padded or trimmed to.

    '''
    if duration_in_samples <= 0:
        raise ScaperError(
            'Duration in samples must be > 0.')
    if not isinstance(duration_in_samples, int):
        raise ScaperError(
            'Duration in samples must be an integer.')

    audio, sr = soundfile.read(audio_path)
    audio_info = soundfile.info(audio_path)
    current_duration = audio.shape[0]

    if duration_in_samples < current_duration:
        audio = audio[:duration_in_samples]
    elif duration_in_samples > current_duration:
        n_pad = duration_in_samples - current_duration

        pad_width = [(0, 0) for _ in range(len(audio.shape))]
        pad_width[0] = (0, n_pad)

        audio = np.pad(audio, pad_width, 'constant')

    soundfile.write(audio_path, audio, sr,
                    subtype=audio_info.subtype, format=audio_info.format)


def peak_normalize(soundscape_audio, event_audio_list):
    """
    Compute the scale factor required to peak normalize the audio such that
    max(abs(soundscape_audio)) = 1.

    Parameters
    ----------
    soundscape_audio : np.ndarray
        The soudnscape audio.
    event_audio_list : list
        List of np.ndarrays containing the audio samples of each isolated
        foreground event.

    Returns
    -------
    scaled_soundscape_audio : np.ndarray
        The peak normalized soundscape audio.
    scaled_event_audio_list : list
        List of np.ndarrays containing the scaled audio samples of
        each isolated foreground event. All events are scaled by scale_factor.
    scale_factor : float
        The scale factor used to peak normalize the soundscape audio.
    """
    eps = 1e-10
    max_sample = np.max(np.abs(soundscape_audio))
    scale_factor = 1.0 / (max_sample + eps)

    # scale the event audio and the soundscape audio:
    scaled_soundscape_audio = soundscape_audio * scale_factor

    scaled_event_audio_list = []
    for event_audio in event_audio_list:
        scaled_event_audio_list.append(event_audio * scale_factor)

    return scaled_soundscape_audio, scaled_event_audio_list, scale_factor


def peak_limiter(soundscape_audio, event_audio_list, samplerate,
                  threshold_db=-0.1, release_ms=50.0, envelope_path=None):
    """
    Apply a peak limiter (pedalboard's lookahead ``Limiter``) to the
    soundscape audio to prevent clipping.

    Parameters
    ----------
    soundscape_audio : np.ndarray
        The soundscape audio.
    event_audio_list : list
        List of np.ndarrays containing the audio samples of each isolated
        foreground event. sum(event_audio_list) must equal
        soundscape_audio (true by construction in scaper).
    samplerate : int
        Sample rate of the audio, required to build the limiter effect.
    threshold_db : float
        Ceiling the limiter holds the signal under, in dBFS. Defaults to
        -0.1, a small safety margin below full scale.
    release_ms : float
        How long (in ms) the limiter takes to let go of a gain reduction
        after a peak has passed. Defaults to 50.
    envelope_path : str or None
        If given, the per-sample gain envelope (same shape/samplerate as
        ``soundscape_audio``, values in (0, 1]) is written to this path as
        a float WAV file, so it can be inspected/plotted alongside the
        mixture (e.g. to sanity-check how aggressively the limiter is
        engaging). Not used anywhere else in scaper. Defaults to None
        (not saved).

    Returns
    -------
    limited_soundscape_audio : np.ndarray
        The soundscape audio after applying the peak limiter.
    limited_event_audio_list : list
        List of np.ndarrays containing the scaled audio samples of each
        isolated foreground event, scaled by the same per-sample gain
        envelope applied to the soundscape.
    scale_factor : float
        The worst-case (minimum) per-sample gain the limiter applied,
        i.e. how much the loudest/most-clipped instant was attenuated.
        1.0 means the limiter never engaged.
    """
    eps = 1e-10

    # Limit the mixture -- and only the mixture, since clipping only
    # exists at the level of the summed signal. pedalboard expects
    # channels-first (n_channels, n_samples), so transpose in and out.
    board = Pedalboard([Limiter(threshold_db=threshold_db, release_ms=release_ms)])
    limited_soundscape_audio = board(
        soundscape_audio.T.astype(np.float32, order='C'),
        samplerate,
    ).T.astype(soundscape_audio.dtype)

    # Recover the per-sample gain the limiter applied, so it can be
    # reapplied identically (and therefore linearly) to each event.
    gain_envelope = np.ones_like(soundscape_audio)
    safe = np.abs(soundscape_audio) > eps  # `True` where amplitdue is not silent.
    gain_envelope[safe] = limited_soundscape_audio[safe] / soundscape_audio[safe]

    if envelope_path is not None:
        soundfile.write(envelope_path, gain_envelope, samplerate, subtype='FLOAT')

    limited_event_audio_list = []
    for event_audio in event_audio_list:
        limited_event_audio_list.append(event_audio * gain_envelope)

    scale_factor = float(gain_envelope.min())

    return limited_soundscape_audio, limited_event_audio_list, scale_factor
