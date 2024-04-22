import tensorflow as tf
import tensorflow_io as tfio

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