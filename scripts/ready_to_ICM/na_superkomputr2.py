# -*- coding: utf-8 -*-
import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import csv
import datetime
from tensorflow.keras import layers, models, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt

#hdf5_file = "container/energy_data.h5"
#test_hdf5_file = "container/angle_data.h5"
hdf5_file = "energy_data.h5"
test_hdf5_file = "angle_data.h5"
p=1
class_coeff=[1.149,1.13]
Energies=np.sort(np.array([130,140,40,30,200,100,90,170,120,150,190,70,160,60,80,180,110,50,25,75,125]))

# Define the generator function
def hdf5_generator(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        data = f['data']  # Assuming data is (x, y, z) for each sample in the file
        labels = f['labels']
        num_samples = data.shape[0]  # The number of samples in the dataset

        for i in range(num_samples):
            image = data[i]  # Shape of image: (x, y, z)
            label = labels[i]  # Shape of label, depending on your task

            # Add color channel dimension to the image (shape becomes (x, y, z, 1))
            image = np.expand_dims(image, axis=-1)  # Adding the color channel (1 for grayscale)
            
            # Yield the image and label
            yield image, label

# Define the dataset pipeline
dataset = tf.data.Dataset.from_generator(
    hdf5_generator,
    args=[hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32),  # Shape with added color channel
        tf.TensorSpec(shape=(), dtype=tf.int16)  # Label shape
    )
)


def test_hdf5_generator(hdf5_file):
    with h5py.File(test_hdf5_file, 'r') as f:
        data = f['data']  # Assuming data is (x, y, z) for each sample in the file
        labels = f['labels']
        num_samples = data.shape[0]  # The number of samples in the dataset

        for i in range(num_samples):
            image = data[i]  # Shape of image: (x, y, z)
            label = labels[i]  # Shape of label, depending on your task

            # Add color channel dimension to the image (shape becomes (x, y, z, 1))
            image = np.expand_dims(image, axis=-1)  # Adding the color channel (1 for grayscale)
            
            # Yield the image and label
            yield image, label

# Define the dataset pipeline
test_dataset = tf.data.Dataset.from_generator(
    test_hdf5_generator,
    args=[test_hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32),  # Shape with added color channel
        tf.TensorSpec(shape=(), dtype=tf.int16)  # Label shape
    )
)

with h5py.File(hdf5_file, "r") as hdf:
    dataset_size = len(hdf["data"])  # Or hdf["labels"], if they have the same length
train_dataset=dataset.take(int(0.8*dataset_size*p))
train_dataset=train_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
val_dataset=test_dataset.take(int(int(0.3*dataset_size*p)))
val_dataset=val_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_ds=test_dataset.skip(int(int(0.3*dataset_size*p))).take(int(int(0.1*dataset_size*p)))
test_ds=test_ds.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_ds2=dataset.skip(int(0.8*dataset_size*p)).take(int(0.2*dataset_size*p))
test_ds2=test_ds2.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=2,                 # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restore model weights from the best epoch
)

class ValLossForE(tf.keras.metrics.Metric):
    def __init__(self, E=5,loss_func=None, name=None, **kwargs):
        # Dynamically set the metric name
        formatted_name = f"loss_for_{0.1 * E}_GeV"
        super().__init__(name=formatted_name, **kwargs)
        if loss_func=="MAE":
            self.loss=tf.keras.losses.mean_absolute_error
        elif loss_func=='mean_squared_error':
            self.loss=tf.keras.losses.mean_squared_error
        else:
            self.loss=loss_func
        self.E = E  # The specific value to filter
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.equal(y_true, self.E)  # Filter only y_true == E

        selected_y_true = tf.boolean_mask(y_true, mask)
        selected_y_pred = tf.boolean_mask(y_pred, mask)

        if tf.size(selected_y_true) > 0:  # Avoid empty tensors
            loss = self.loss(selected_y_true, selected_y_pred)
            self.total_loss.assign_add(tf.reduce_sum(loss))
            self.count.assign_add(tf.cast(tf.size(loss), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total_loss, self.count)  # Avoid division by zero

def snorm_RMS(y_true, y_pred):
    """Square Root Normalized Root Mean Square (RMS) divided by sqrt(y_true)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = tf.sqrt(rms) / tf.maximum(tf.sqrt(y_true), 1e-10)  # Avoid division by zero
    return tf.reduce_mean(snorm_rms, axis=-1)

def SaveModel(a,b,c,d, filename,model):
    combined_array = np.vstack((a, b, c, d)).T

    # Step 2: Define the header for the CSV file
    header = ['ME', 'MAE', 'normMAE', 'RMS']

    # Step 3: Write the combined array to a CSV file
    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(header)
        # Write the data
        writer.writerows(combined_array)  
    model.save('model+'+filename)

def Metrics(E_true,E_pred,Energies):
    RMS=np.zeros((len(Energies)))
    ME=np.zeros((len(Energies)))
    MAE=np.zeros((len(Energies)))
    for i in range(0,len(Energies)):
        RMS[i]=np.sqrt(np.mean((E_true[E_true==Energies[i]]-E_pred[E_true==Energies[i]])**2))
        ME[i]=np.mean(E_pred[E_true==Energies[i]]-E_true[E_true==Energies[i]])
        MAE[i]=np.mean(np.abs(E_pred[E_true==Energies[i]]-E_true[E_true==Energies[i]]))
    return RMS, ME, MAE, MAE/np.sqrt(Energies)

#LISTA MODELI
Models=[]
#TUTAJ DEFINIUJE WIĘKSZY MODEL
#Tu zdefiniuj model
input_3d = Input(shape=(20, 110, 11, 1), name='3D_Input')

# 3D CNN Path
x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(input_3d)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

# Flatten 3D features
x = layers.Flatten()(x)

summed_tensor = Lambda(lambda t: tf.reduce_sum(t, axis=[1, 2, 3, 4]))(input_3d)  # Sum over spatial dimensions
summed_tensor_expanded = Lambda(lambda t: tf.expand_dims(t, axis=-1))(summed_tensor)  # Expand to (None, 1)
summed_tensor_exp =(summed_tensor_expanded/10-class_coeff[1])/class_coeff[0]
# Ensure shapes are compatible for concatenation
combined = layers.Concatenate(axis=-1)([x, summed_tensor_expanded])  # Concatenate along last axis


# Fully Connected Layers
fc = layers.Dense(64, activation='relu')(combined)
fc = layers.Dense(32, activation='relu')(fc)
fc = layers.Dense(16, activation='relu')(fc)


# Output Layer
output = layers.Dense(1, activation='linear')(fc)

# Define the Model
models.append(Model(inputs=[input_3d], outputs=output))

#TUTAJ DEFINIUJE MNIEJSZY MODEL
input_3d = Input(shape=(20, 110, 11, 1), name='3D_Input')

# 3D CNN Path
x = layers.Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(input_3d)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = layers.Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

# Flatten 3D features
x = layers.Flatten()(x)

summed_tensor = Lambda(lambda t: tf.reduce_sum(t, axis=[1, 2, 3, 4]))(input_3d)  # Sum over spatial dimensions
summed_tensor_expanded = Lambda(lambda t: tf.expand_dims(t, axis=-1))(summed_tensor)  # Expand to (None, 1)
summed_tensor_exp =(summed_tensor_expanded/10-class_coeff[1])/class_coeff[0]
# Ensure shapes are compatible for concatenation
combined = layers.Concatenate(axis=-1)([x, summed_tensor_expanded])  # Concatenate along last axis


# Fully Connected Layers
fc = layers.Dense(32, activation='relu')(combined)
fc = layers.Dense(16, activation='relu')(fc)
fc = layers.Dense(8, activation='relu')(fc)


# Output Layer
output = layers.Dense(1, activation='linear')(fc)

models.append(Model(inputs=[input_3d], outputs=output))
nazwy=['Bigger','Smaller']
logs = "logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logs,
                                                 histogram_freq=1,
                                                 profile_batch='420,430')
for model, nazwa in zip(models,nazwy):
    metrics=[]
    for E in Energies:
        metrics.append(ValLossForE(E=E,loss_func=snorm_RMS))
    model.compile(optimizer='adam', loss=snorm_RMS, metrics=metrics) 
    csv_logger = tf.keras.callbacks.CSVLogger(f"training_log_{nazwa}.csv", append=True)

    with tf.device('/GPU:0'):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            shuffle=True,
            epochs=10,                 # Set a high number of epochs; EarlyStopping will stop early
            callbacks=[early_stopping, csv_logger],  # Add EarlyStopping to callbacks
            verbose=1
        )
    
    #Generowanie danych
    with tf.device('/GPU:0'):
        y_pred=model.predict(test_ds).flatten()
        y_true = np.array([label.numpy() for _, label in test_ds]).flatten()
        y_pred2=model.predict(test_ds2).flatten()
        y_true2 = np.array([label.numpy() for _, label in test_ds2]).flatten()
        y_true=np.concatenate((y_true,y_true2),axis=0)
        y_pred=np.concatenate((y_pred,y_pred2),axis=0)
    RMS, ME, MAE, normMAE = Metrics(y_true, y_pred, Energies)
    SaveModel(ME,MAE,normMAE, RMS,nazwa +'.csv',model )
