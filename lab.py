# %%
import tensorflow as tf
from utils import MetricsLogger
import os

from dataset import get_datasets, labels_v1, labels_v2
from stormer import Stormer

# %%
#### hyper parameters that defines the structure of the model
version = 2
num_classes = len(labels_v1) if version == 1 else len(labels_v2)

learning_rate =     0.000005
weight_decay =      0.000005
batch_size =        128
num_epochs = 10000  # For real training, use num_epochs=100. 10 is a test value
# patch_size = 6  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2

num_heads = 4
num_repeats = 2
num_state_cells = [10, 10]
input_seq_size = 31
projection_dim = 32
inner_ff_dim = 2 * projection_dim
dropout = 0.1
probability_of_noise = 0.8


# %%
dataset_type = 'mel'
train, valid, test = get_datasets(
    batch_size=batch_size,
    type=dataset_type,
    probability_of_noise=probability_of_noise,
    version=version
)

# %%
stormer = Stormer(
    num_classes=num_classes,
    num_repeats=num_repeats,
    num_heads=num_heads,
    num_state_cells=num_state_cells,
    input_seq_size=input_seq_size,
    projection_dim=projection_dim,
    inner_ff_dim=inner_ff_dim,
    dropout=dropout,
    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
)

# %%
# load the model weights
model_path =f"./models/stormer_r{num_repeats}_h{num_heads}_dm{projection_dim}_dataset={dataset_type}/stormer_r{num_repeats}_h{num_heads}_dm{projection_dim}_dataset={dataset_type}.ckpt"
# check if the model containing directory exists
load_weights = os.path.exists(os.path.dirname(model_path))
if load_weights:
    stormer.load_weights(model_path)

# %%
results_filename = f'data/results/results_r{num_repeats}_h{num_heads}_dm{projection_dim}_dataset={dataset_type}.csv'

metrics=["accuracy"]

stormer.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate),
    loss="categorical_crossentropy",
    metrics=metrics,
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

state_transformer_history = stormer.fit(
    train,
    validation_data=valid,
    epochs=num_epochs,
    callbacks=[
        model_checkpoint_callback,
        MetricsLogger(
            results_filename,
        )
    ],
)
