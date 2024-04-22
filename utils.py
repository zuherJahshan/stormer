import tensorflow as tf
from tensorflow import keras
import csv
import os
import json
from typing import Dict, Any


class MetricsLogger(keras.callbacks.Callback):
    def __init__(self, log_file=None):
        super(MetricsLogger, self).__init__()
        self.log_file = log_file
        if log_file is not None:
            # Open the log file in append mode
            # get the number of lines written to the file
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    line_count = sum(1 for line in f)
            else:
                line_count = 0
            self.new_file = line_count == 0
            self.file = open(log_file, 'a')
            self.writer = csv.writer(self.file)
        else:
            self.file = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.new_file:
            self.writer.writerow(list(logs.keys()))
            self.new_file = False
        # Optionally write to a file
        if self.file is not None:
            self.writer.writerow(list(logs.values()))
            self.file.flush()
            

def transfer(
    from_model: tf.keras.Model,
    to_model: tf.keras.Model,
):
    transfered_layers = []
    for i, from_layer in enumerate(from_model.layers):
        # check if the same kind of layer as the i'th layer of the current model
        if isinstance(from_layer, to_model.layers[i].__class__):
            # transfer weights
            # check if the weights have the same shape
            if len(from_layer.get_weights()) > 0 and \
                not from_layer.get_weights()[0].shape == to_model.layers[i].get_weights()[0].shape:
                break

            to_model.layers[i].set_weights(from_layer.get_weights())
            transfered_layers.append(from_layer.name)
        else:
            break
    return to_model, transfered_layers


def schedule(epoch, lr):
    drop_rate = 0.8
    epochs_drop = 10
    if epoch % epochs_drop == 0:
        return lr * drop_rate
    return lr


lrs = tf.keras.callbacks.LearningRateScheduler(
    schedule,
    verbose=0
)


def save_hps(
    model_name,
    hps: Dict[str, Any],
):
    os.makedirs('models/hps/', exist_ok=True)
    with open(f'models/hps/{model_name}.json', 'w') as f:
        json.dump(hps, f)


def load_hps(model_name):
    with open(f'models/hps/{model_name}.json', 'r') as f:
        return json.load(f)
    

def get_model_name(
    num_repeats,
    num_heads,
    projection_dim,
    dataset_type,
    **kwargs,
):
    return f"stormer_r{num_repeats}_h{num_heads}_dm{projection_dim}_dataset={dataset_type}"


def get_model_path(model_name):
    return f"./models/{model_name}/{model_name}.ckpt"


def print_enumerated_list(list_of_items, item_name="item"):
    if not list_of_items:
        print(f"No {item_name}s available.", flush=True)
        return
    
    # Determine the maximum length of model names for proper alignment
    max_name_length = max(len(name) for name in list_of_items)
    
    # Set minimum column width for aesthetics
    min_column_width = 15
    name_column_width = max(max_name_length, min_column_width)
    
    # Calculate the width of the table dynamically based on content
    index_column_width = 12  # Fixed width for the index column
    separator_length = 2 + index_column_width + 3 + name_column_width + 2  # Calculation of total separator length

    # Print header
    print(f"Available {item_name}s:", flush=True)
    print("-" * separator_length, flush=True)  # Adjust the total length of the separator line
    header_format = "| {0:<{index_width}} | {1:<{name_width}} |".format("Index", "Model Name", index_width=index_column_width, name_width=name_column_width)
    print(header_format)
    print("|" + "-" * (index_column_width + 2) + "|" + "-" * (name_column_width + 2) + "|")
    
    # Print each model name with its index
    for index, name in enumerate(list_of_items):
        row_format = f"| {index:<{index_column_width}} | {name:<{name_column_width}} |"
        print(row_format)
    
    # Print closing line for the table
    print("|" + "_" * (index_column_width + 2) + "|" + "_" * (name_column_width + 2) + "|", flush=True)