import tensorflow_datasets as tfds
import tensorflow as tf
import math

from urban_sound_ds import get_background_noise_dataset
from speech_commands_v2_ds import get_datasets as sc_get_datasets

FREQUENCY = 16_000
DURATION = FREQUENCY # Which means 2 seconds


def get_time_steps(frame_length, frame_step, duration=DURATION):
    return int(math.ceil((duration - frame_length + 1) / frame_step))


def get_audio_and_label(x, noise, version):
    audio, label = x
    
    return audio, noise, label


# Change audio to spectrogram and label to one-hot encoded label
def get_spectogram(audio, label, version, frame_length, frame_step):
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)

    return spectrogram, label


def get_mel_spectogram(spectrogram, label, mel_bands):
    lower_edge_hertz, upper_edge_hertz = 0, FREQUENCY // 2
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bands,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=FREQUENCY,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    # mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return mel_spectrogram, label


def compute_deltas(mfccs):
    # Pad the MFCCs at the beginning and end along the time dimension (axis 1)
    padded_mfccs = tf.pad(mfccs, paddings=[[1, 1], [0, 0]], mode='SYMMETRIC')
    # Compute the deltas (first-order differences)
    deltas = padded_mfccs[2:, :] - padded_mfccs[:-2, :]
    return deltas / 2.0


def get_mfccs(mel_spectrogram, label, num_coefficients=13):
    # add delta and delta-delta
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    deltas = compute_deltas(mfccs)
    delta_deltas = compute_deltas(deltas)
    # Concatenate along the last dimension
    mfccs = tf.concat([mfccs, deltas, delta_deltas], axis=-1)
    return mfccs, label


def insure_falling_between_1_and_m1(example, label):
    max_val = tf.reduce_max(tf.abs(example), axis=-1)
    return example / max_val, label




def convert_to_ragged(example, label):
    return tf.RaggedTensor.from_tensor(example, padding=0), label


def get_last_dimension(type, num_coefficients, mel_bands, frame_length):
    if type == "spec":
        return frame_length // 2 + 1
    if type == "mel":
        return mel_bands
    if type == "mfccs":
        return num_coefficients * 3


def convert_to_tensor(example_ragged, label, num_coefficients, mel_bands, frame_length, frame_step, type):
    # Determine the last dimension size based on the type of features
    last_dim_size = get_last_dimension(type, num_coefficients, mel_bands, frame_length)
    
    # Convert the ragged tensor to a fixed-size tensor
    # The shape argument is set to [None, None, last_dim_size] which means:
    # - Keep the existing sizes for the first two dimensions (batch and time steps)
    # - Set the last dimension to the calculated last_dim_size
    example_tensor = example_ragged.to_tensor(shape=[None, get_time_steps(frame_length, frame_step), last_dim_size])

    return example_tensor, label



def add_time_shift_noise_and_align(audio, noise, label, max_shift_in_ms):
    # randomly shift the audio by at most 100ms
    max_shift = (max_shift_in_ms * FREQUENCY) // 1000
    time_shift = tf.random.uniform(shape=(), minval=0, maxval=max_shift, dtype=tf.int32)
    future = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)

    audio = tf.cond(
        future == 0,
        lambda: tf.pad(audio[time_shift:], paddings=[[0, time_shift]]),
        lambda: tf.pad(audio[:-time_shift], paddings=[[time_shift, 0]])
    )
    
    if tf.shape(audio)[0] < DURATION:
        audio = tf.pad(audio, paddings=[[DURATION - tf.shape(audio)[0], 0]])

    noise_redution_coefficient = 8
    audio = (audio[:DURATION] + (noise[:DURATION] / noise_redution_coefficient)) / 2
    audio.set_shape((DURATION,))

    return audio, label


def get_tf_dataset(
    raw_dataset,
    frame_length,
    frame_step,
    version,
    type,
    batch_size,
    mel_bands,
    num_coefficients,
    max_shift_in_ms
):
    # Always happens
    ds = raw_dataset.map(
        lambda x, noise: get_audio_and_label(x, noise, version),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    # Data augmentation
    ds = ds.map(
        lambda audio, noise, label: add_time_shift_noise_and_align(audio, noise, label, max_shift_in_ms),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    # Transform to spectogram
    
    ds = ds.map(
        lambda audio, label: get_spectogram(audio, label, version, frame_length, frame_step),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    # Convert to mel spectograms
    if type != "spec":
        ds = ds.map(
            lambda spectrogram, label: get_mel_spectogram(spectrogram, label, mel_bands),
            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Convert to MFCCs
    if type != "mel":
        ds = ds.map(
            lambda mel, label: get_mfccs(mel, label, num_coefficients),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.\
        shuffle(4096).\
        map(convert_to_ragged, num_parallel_calls=tf.data.AUTOTUNE).\
        ragged_batch(batch_size).\
        map(lambda example, label: convert_to_tensor(example, label, num_coefficients, mel_bands, frame_length, frame_step, type), num_parallel_calls=tf.data.AUTOTUNE).\
        prefetch(tf.data.AUTOTUNE)



def get_dataset_shape(
    frame_length=256,
    frame_step=128,
    dataset_type="mfccs",
    mel_bands=40,
    num_coefficients=13
):
    time_steps = get_time_steps(frame_length, frame_step)
    last_dim_size = get_last_dimension(dataset_type, num_coefficients, mel_bands, frame_length)
    return (1, time_steps, last_dim_size)


# Will return three tf.data.Dataset objects, one for each split (train, validation, test)
def get_datasets(
    batch_size,
    frame_length=256,
    frame_step=128,
    version=1, # Could be 1 or 2
    dataset_type="mfccs", # Could be spec, mel, or mfcc
    mel_bands=40,
    num_coefficients=13,
    max_shift_in_ms=100,
    probability_of_noise=1.0,
    **kwargs
):
    print(f"Loading dataset version {version}")
    combined_ds = sc_get_datasets()
    print(f"Dataset loaded")
    return (
        get_tf_dataset(
            tf.data.Dataset.zip(
                combined_ds["train"],
                get_background_noise_dataset(
                    "data/urban_sound",
                    combined_ds["train"].cardinality().numpy(),
                    probability_of_noise=probability_of_noise
                )
            ),
            frame_length,
            frame_step,
            version,
            dataset_type,
            batch_size,
            mel_bands,
            num_coefficients,
            max_shift_in_ms
        ),
        get_tf_dataset(
            tf.data.Dataset.zip(
                combined_ds["valid"],
                get_background_noise_dataset(
                    "data/urban_sound",
                    combined_ds["valid"].cardinality().numpy(),
                    probability_of_noise=0
                )
            ),
            frame_length,
            frame_step,
            version,
            dataset_type,
            batch_size,
            mel_bands,
            num_coefficients,
            max_shift_in_ms
        ),
        get_tf_dataset(
            tf.data.Dataset.zip(
                combined_ds["test"],
                get_background_noise_dataset(
                    "data/urban_sound",
                    combined_ds["test"].cardinality().numpy(),
                    probability_of_noise=0
                )
            ),
            frame_length,
            frame_step,
            version,
            dataset_type,
            batch_size,
            mel_bands,
            num_coefficients,
            max_shift_in_ms
        ),
    )