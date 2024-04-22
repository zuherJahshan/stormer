# %%
import tensorflow as tf
import utils
import os
import glob

from dataset import get_datasets, get_dataset_shape
from stormer import Stormer

# %%
## Example of how you can build a new model and transfer learn from an old model
import copy

print_model_table = lambda model_list: utils.print_enumerated_list(model_list, "Model")

models_names = [path.split("/")[-1] for path in glob.glob("models/stormer*")]
models_names.sort()
print_model_table(models_names)

old_model_name = models_names[0]

# %%
old_hps = utils.load_hps(old_model_name)
new_hps = copy.deepcopy(old_hps)
new_num_state_cells = [1]
new_hps["num_state_cells"] += new_num_state_cells
new_hps["num_repeats"] += len(new_num_state_cells)
new_model_name = utils.get_model_name(**new_hps)

utils.save_hps(new_model_name, new_hps)

# %%
old_stormer = Stormer(**old_hps)
new_stormer = Stormer(**new_hps)

old_stormer.load_weights(utils.get_model_path(old_model_name))

stormer, transfered_layers = utils.transfer(old_stormer, new_stormer)

# %%
for i in range(len(transfered_layers)):
    stormer.layers[i].trainable = False

# %%
## load the datasets
train, valid, test = get_datasets(**new_hps)

# %%
results_filename = f'data/results/{new_model_name}.csv'

metrics=["accuracy"]

new_stormer.compile(
    optimizer=tf.keras.optimizers.AdamW(new_hps["learning_rate"]),
    loss="categorical_crossentropy",
    metrics=metrics,
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=utils.get_model_path(new_model_name),
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

state_transformer_history = new_stormer.fit(
    train,
    validation_data=valid,
    epochs=new_hps["num_epochs"],
    callbacks=[
        model_checkpoint_callback,
        utils.MetricsLogger(
            results_filename,
        )
    ],
)

# %%
stormer.summary()

# %%



