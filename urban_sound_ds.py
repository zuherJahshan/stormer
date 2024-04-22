import tensorflow as tf
import glob
import ds_utils

sampling_rate = 44100


def get_background_noise_dataset(datapath, size, probability_of_noise):
    noises = glob.glob(f"{datapath}/*.wav")
    # If the size of noises is less than the required size, we will repeat the noises
    new_noises = []
    for i in range(size):

        new_noises.append(noises[i % len(noises)])
    files_ds = tf.data.Dataset.from_tensor_slices(new_noises)
    audio_ds = files_ds.map(
        ds_utils.get_audio_from_wav_file,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    adjusted_audio_ds = audio_ds.map(
        lambda x: ds_utils.adjust_frequncy(x, 16000, 44100),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    fixed_length_audio_ds = adjusted_audio_ds.map(
        lambda x: ds_utils.pad_and_align(x, 16000),
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
