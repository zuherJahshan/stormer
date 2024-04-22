import random
import glob
import os
import ds_utils
import tensorflow as tf

# datapath should point to the absolute path
datapath = "data/speech_commands_v2"
datapath = os.path.abspath(datapath)


def divide(
    datapath,
    train_portion = 0.9,
    valid_portion = 0.05
    ):
    train = [] # a list of tuples, the first is the sample, the second is the label
    valid = []
    test = []
    for label in os.listdir(datapath):
        sample_files = glob.glob(f"{datapath}/{label}/*.wav")
        sample_files.sort()
        train_size = int(len(sample_files) * train_portion)
        valid_size = int(len(sample_files) * valid_portion)
        test_size = len(sample_files) - train_size - valid_size
        train.extend([(sample, label) for sample in sample_files[:train_size]])
        valid.extend([(sample, label) for sample in sample_files[train_size:train_size+valid_size]])
        test.extend([(sample, label) for sample in sample_files[train_size+valid_size:]])
        
    random.shuffle(train)
    random.shuffle(valid)
    random.shuffle(test)
    return {
        "train": train,
        "valid": valid,
        "test": test
    }


labels_to_indices = {}
labels = os.listdir(datapath)
for label in labels:
    if label.startswith("."):
        labels.remove(label) ## TODO: change it to ckeck if file or directory
labels.sort()
num_of_labels = len(labels)
for i, label in enumerate(labels):
    labels_to_indices[label] = i


def get_dataset_from_list(sample_file_and_labels_list):
    # split to to two lists
    sample_files, labels = zip(*sample_file_and_labels_list)
    sample_files = list(sample_files)
    labels = list(labels)
    labels_idxs = [labels_to_indices[label] for label in labels]
    
    # generate the sample files ds and the labels ds
    files_ds = tf.data.Dataset.from_tensor_slices(sample_files)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels_idxs)
    labels_ds = labels_ds.map(
        lambda x: tf.one_hot(x, num_of_labels)
    )
    audio_ds = files_ds.map(
        ds_utils.get_audio_from_wav_file,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # adjusted_audio_ds = audio_ds.map(
    #     lambda x: adjust_frequncy(x, 16000, 44100),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE
    # )
    fixed_length_audio_ds = audio_ds.map(
        lambda x: ds_utils.pad_and_align(x, 16000),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # zip the audio and labels together
    return tf.data.Dataset.zip((fixed_length_audio_ds, labels_ds)).shuffle(1024)


def get_datasets():
    data = divide(datapath)
    return {
        "train": get_dataset_from_list(data["train"]),
        "valid": get_dataset_from_list(data["valid"]),
        "test": get_dataset_from_list(data["test"])
    }
