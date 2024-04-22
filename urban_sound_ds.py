import tensorflow as tf
import glob
import tensorflow_io as tfio

sampling_rate = 44100
def get_audio_from_wav_file(file):
    audio, sample_rate = tf.audio.decode_wav(
        tf.io.read_file(file),
        desired_channels=1
    )
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

def adjust_frequncy(audio, new_sampling_rate=16000, old_sampling_rate=44100):
    audio = tf.cast(audio, tf.float32)
    audio = tfio.audio.resample(
        audio,
        rate_in=old_sampling_rate,
        rate_out=new_sampling_rate
    )
    return audio


def pad_and_align(audio, target_length=16000):
    if tf.shape(audio)[0] > target_length:
        starting_idx = tf.random.uniform(
            shape=(),
            minval=0,
            maxval=tf.shape(audio)[0] - target_length,
            dtype=tf.int32
        )
        audio = audio[starting_idx: starting_idx + target_length]
    else:
        audio = tf.concat([audio, tf.zeros(target_length - tf.shape(audio)[0], dtype=tf.float32)], axis=0)
        # ensure the audio is in the correct shape
        audio.set_shape((target_length,))
    return audio


def get_background_noise_dataset(datapath, size, probability_of_noise):
    noises = glob.glob(f"{datapath}/*.wav")
    # If the size of noises is less than the required size, we will repeat the noises
    new_noises = []
    for i in range(size):

        new_noises.append(noises[i % len(noises)])
    files_ds = tf.data.Dataset.from_tensor_slices(new_noises)
    audio_ds = files_ds.map(
        get_audio_from_wav_file,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    adjusted_audio_ds = audio_ds.map(
        lambda x: adjust_frequncy(x, 16000, 44100),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    fixed_length_audio_ds = adjusted_audio_ds.map(
        lambda x: pad_and_align(x, 16000),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    sparse_noise_ds = fixed_length_audio_ds.map(
        lambda x: tf.cond(
            tf.random.uniform(shape=()) < probability_of_noise,
            lambda: x,
            lambda: tf.zeros_like(x)
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return sparse_noise_ds.shuffle(1024)
