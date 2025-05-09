# %%
import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow as tf
import numpy as np
import math
import csv
import time
from tensorflow.keras import layers, models, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from datetime import datetime

# %%
hdf5_file = "angles_data_w_labels.h5"
#klasyka_file='klasyka_z_regresja.csv' #TU CZEKAM NA KOD OD PATRYCJI
#klasyka_file2='TE_RMSE_sigma.txt'

# %%
p=1
class_coeff=[1.149,1.13]
Energies=np.sort(np.array([130,140,40,30,200,100,90,170,120,150,190,70,160,60,80,180,110,50,25,75,125]))

# %%
# Define the generator function
def hdf5_generator(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        data = f['data']  # Assuming data is (x, y, z) for each sample in the file
        labels = f['labels']
        shifts = f['shift']
        sum_energies = f['energy']
        num_samples = data.shape[0]  # The number of samples in the dataset
        """
        print(f"Number of samples in the dataset: {num_samples}")
        print(f"Shape of data: {data.shape}")
        print(f"Shape of labels: {labels.shape}")
        print(f"Shape of shift: {shifts.shape}")"""

        for i in range(num_samples):
            image = data[i]  # Shape of image: (x, y, z)
            label = labels[i].astype(np.float32).reshape(1)  # Shape of label, depending on your task
            shift = shifts[i,:].astype(np.int16)
            sum_energy = sum_energies[i].astype(np.float32).reshape(1)  # Convert to float32 for TensorFlow compatibility
            # Add color channel dimension to the image (shape becomes (x, y, z, 1))
            image = np.expand_dims(image, axis=-1)  # Adding the color channel (1 for grayscale)
            # Yield the image and label
            yield image, shift

# Define the dataset pipeline
dataset = tf.data.Dataset.from_generator(
    hdf5_generator,
    args=[hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32, name='x'),
        tf.TensorSpec(shape=(4,), dtype=tf.int16,name = 'y')  # Sum energy shape
    )
)


# %%
with h5py.File(hdf5_file, "r") as hdf:
    dataset_size = len(hdf["data"])  # Or hdf["labels"], if they have the same length
    print(f"Dataset size: {dataset_size}")



# %%
p=1
batchsize = 32
#Podzielmy datasety na treningowy, walidacyjny i testowy
train_dataset=dataset.take(int(2*dataset_size*p/3))
train_dataset=train_dataset.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_dataset = dataset.skip(int(2*dataset_size*p/3)).take(int(dataset_size*p/3))

val_dataset=test_dataset.take(int(int(0.3*dataset_size*p)))
val_dataset=val_dataset.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds=test_dataset.skip(int(int(0.3*dataset_size*p))).take(int(int(0.1*dataset_size*p)))
test_ds=test_ds.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print("Dane treningowe, walidacyjne i testowe ready")

# %%
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=3,                 # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restore model weights from the best epoch
)

# %%
#Tu zdefiniuj model
input_3d = Input(shape=(20, 110, 11, 1,), name='3D_Input')

# 3D CNN Path
x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(input_3d)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

# Flatten 3D features
x = layers.Flatten()(x)

# Fully Connected Layers
fc = layers.Dense(64, activation='relu')(x)
fc = layers.Dense(32, activation='relu')(fc)
fc = layers.Dense(16, activation='relu')(fc)

output = layers.Dense(4, activation='linear')(fc)


# Define the Model Important to have two imputs
model = Model(inputs=[input_3d], outputs=output)

#compile the model with no metrics for now
model.compile(optimizer='adam',
              loss='mean_squared_error')
            
model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

csv_logger = tf.keras.callbacks.CSVLogger("training_log5.csv", append=True)

logs = "logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logs,
                                                 histogram_freq=1,
                                                 profile_batch='420,430')


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    shuffle=True,
    epochs=1,                 # Set a high number of epochs; EarlyStopping will stop early
    callbacks=[early_stopping,tboard_callback],  # Add EarlyStopping to callbacks
    verbose=1
    )

model.save('test_angles.h5')
