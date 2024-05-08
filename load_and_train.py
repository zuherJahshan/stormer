# %%
import tensorflow as tf
import utils
import os
import glob

from dataset import get_datasets
from gated_stormer import Stormer

# %%
# print model names

print_model_table = lambda model_list: utils.print_enumerated_list(model_list, "Model")

models_names = [path.split("/")[-1] for path in glob.glob("models/*stormer*")]
models_names.sort()
print_model_table(models_names)
model_name = models_names[0]

# %%
hps = utils.load_hps(model_name)
# Change HYPER PARAMETERS
# hps["weight_decay"] /= 50

utils.save_hps(model_name, hps)
stormer = Stormer(**hps)

# %%
## load the datasets
train, valid, test = get_datasets(
    **hps
)

# %%
# Build the stormer model
for example, _ in train.take(1):
    stormer(example)

# %%
# check if the model containing directory exists
model_path = utils.get_model_path(model_name)
load_weights = os.path.exists(os.path.dirname(model_path))
if load_weights:
    stormer.load_weights(model_path)

# %%
results_filename = f'data/results/{model_name}.csv'

metrics=["accuracy"]

stormer.compile(
    optimizer=tf.keras.optimizers.AdamW(hps["learning_rate"]),
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
    epochs=hps["num_epochs"],
    callbacks=[
        model_checkpoint_callback,
        utils.MetricsLogger(
            results_filename,
        )
    ],
)


