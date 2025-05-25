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
hdf5_file = "angles_data3.h5"
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
        shifts = f['shift_labels']
        sum_energies = f['energy']
        num_samples = data.shape[0]  # The number of samples in the dataset
        classical_shifts = f["shifts"]
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
            classical_shift = classical_shifts[i,:].astype(np.float32)
            # Add color channel dimension to the image (shape becomes (x, y, z, 1))
            image = np.expand_dims(image, axis=-1)  # Adding the color channel (1 for grayscale)
            # Yield the image and label
            yield (image, sum_energy, classical_shift), shift

# Define the dataset pipeline
dataset = tf.data.Dataset.from_generator(
    hdf5_generator,
    args=[hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        (tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32, name='x'),
         tf.TensorSpec(shape=(1,), dtype=tf.float32, name='sum_energy'),
         tf.TensorSpec(shape=(4,), dtype=tf.float32, name='classical_shift')),                       
                    
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
train_dataset=dataset.take(int(4*dataset_size*p/9))
train_dataset=train_dataset.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_dataset = dataset.skip(int(4*dataset_size*p/9)).take(int(5*dataset_size*p/9))

val_dataset=test_dataset.take(int(4*dataset_size*p/9))
val_dataset=val_dataset.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds=test_dataset.skip(int(4*dataset_size*p/9)).take(int(dataset_size*p/9))
test_ds=test_ds.batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print("Dane treningowe, walidacyjne i testowe ready")

# %%
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=3,                 # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restore model weights from the best epoch
)

# %%
def GetModel(convFilters:np.array,denseNeurons:np.array,LossFunction="mean_squared_error"):
    """
    Function to create a 3D CNN model with two inputs: 3D data and energies.

    - convFiltesr: np.array of integers defining the number of filters in each convolutional layer.
    - denseNeurons: np.array of integers defining the number of neurons in each dense layer.
    
    can add arguments for regularization, activation functions, kernel size
    """
    # Input Layer
    input_3d = Input(shape=(20, 110, 11, 1,), name='3D_Input')
    input_classical_shift = Input(shape=(4,), name='Classical_Shift_Input')
    input_sum_energy = Input(shape=(1,), name='Sum_Energy_Input')
    
    # 3D CNN Path
    x = input_3d
    for nFeatures in convFilters:
        x = layers.Conv3D(nFeatures, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Flatten 3D features
    x = layers.Flatten()(x)

    combined = layers.Concatenate(axis=-1)([x, input_classical_shift, input_sum_energy])  # Concatenate along last axis

    # Fully Connected Layers
    x = combined
    for nNeurons in denseNeurons:
        x = layers.Dense(nNeurons, activation='relu')(x)

    # Output Layer
    output = layers.Dense(4, activation='linear')(x)

    # Define the Model 
    model = Model(inputs=[input_3d,input_sum_energy,input_classical_shift], outputs=output)

    #compile the model with no metrics for now
    model.compile(optimizer='adam',
                loss=LossFunction)
    return model

# %%
def SaveModel(a,b,true_shift, filename,model):
    combined_array = np.hstack((true_shift,a, b))

    # Step 2: Define the header for the CSV file
    header = ['True_xShift','True_yShift','True_xAngle','True_yAngle',
              'ME xShift','ME yShift','ME xAngle', 'ME yAngle',
              'RMS xShift','RMS yShift','RMS xAngle','RMS yAngle',]

    # Step 3: Write the combined array to a CSV file
    with open(filename+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(header)
        # Write the data
        writer.writerows(combined_array)  
    model.save('model+'+filename+".h5")

def Metrics(shift_true,shift_pred):
    #shift_ true = [xShift, yShift, xAngle, yAngle]
    RMS = (np.sqrt((shift_true - shift_pred) ** 2))
    ME = shift_true - shift_pred
    #MAE = np.abs(shift_true - shift_pred)
    return RMS, ME

# %%
# DEFINE THE MODEL
models,names = [],[]

# EXAMPLE MODEL
models.append(GetModel(convFilters=[16,32,64],denseNeurons=[64,32,16],LossFunction="mse"))
names.append("conv_adcending")
models.append(GetModel(convFilters=[64,32,16],denseNeurons=[64,32,16],LossFunction="mse"))
names.append("conv_descending")

# %%
# Train the models
for model, name in zip(models,names):
    try:
        os.mkdir(f"{name}_logs")
    except FileExistsError:
        pass
    logs = f"{name}_logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logs,
                                                    histogram_freq=1,
                                                    profile_batch='420,430')
    with tf.device('/GPU:0'):
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1,
            callbacks=[tboard_callback],
            verbose=0
            )
        #Generowanie danych
        y_pred = model.predict(test_ds)
        y_true = np.array([label.numpy() for _, label in test_ds]).reshape(-1, 4)

    print(f"Saving model {name}")
    RMS, ME= Metrics(y_true, y_pred)
    SaveModel(ME,RMS,y_true,name, model)

    # Saving model history to CSV
    combined_array = np.vstack(list(history.history.values())).T
    header = list(history.history.keys())
    with open(f"{name}_history"+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(combined_array)


