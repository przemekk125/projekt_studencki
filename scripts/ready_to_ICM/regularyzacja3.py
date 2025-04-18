# model1
# reg_conv = regularizers.l2(1e-4)
# reg_dense = regularizers.L1L2(l1=1e-5, l2=1e-3)

# model2
# reg_conv = regularizers.l2(1e-4)
# reg_dense = regularizers.L1L2(l1=1e-3, l2=1e-2)

# model3
reg_conv = regularizers.l2(1e-3)
reg_dense = regularizers.L1L2(l1=1e-5, l2=1e-3)

# model4
# reg_conv = regularizers.l2(1e-4)
# reg_dense = regularizers.L2(1e-3)

# model5
# reg_conv = regularizers.l2(1e-4)
# reg_dense = regularizers.L1(1e-4)

import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.config.set_soft_device_placement(True)
import numpy as np
import csv
from tensorflow.keras import layers, models, Model, Input, regularizers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Lambda # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from datetime import datetime

"""     GLOBALS     """
# hdf5_file and test_hdf5_file only for determining the size of the dataset
# they are not used in the training process
hdf5_file = "/home/pkaleta/data/energy_data.h5"
test_hdf5_file = "/home/pkaleta/data/angles_data.h5"
TrainDsPath = "/home/pkaleta/data/tf_train_dataset"
TestDsPath = "/home/pkaleta/data/tf_test_dataset"


p=1
Energies=np.sort(np.array([130,140,40,30,200,100,90,170,120,150,190,70,160,60,80,180,110,50,25,75,125]))

""" FUNCTIONS       """

def snorm_MSE(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = rms / tf.maximum(y_true , 1e-10)  # Avoid division by zero
    return tf.reduce_mean(snorm_rms, axis=-1)


def GetModel(convFilters:np.array,denseNeurons:np.array,LossFunction=snorm_MSE):
    """
    Function to create a 3D CNN model with two inputs: 3D data and energies.

    - convFiltesr: np.array of integers defining the number of filters in each convolutional layer.
    - denseNeurons: np.array of integers defining the number of neurons in each dense layer.
    
    can add arguments for regularization, activation functions, kernel size
    """
    # Input Layer
    input_3d = Input(shape=(20, 110, 11, 1,), name='3D_Input')
    input_energies = Input(shape=(1,), name='Energies_Input')
    
    # 3D CNN Path
    x = input_3d
    for nFeatures in convFilters:
        x = layers.Conv3D(nFeatures, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=reg_conv)(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Flatten 3D features
    x = layers.Flatten()(x)

    combined = layers.Concatenate(axis=-1)([x, input_energies])  # Concatenate along last axis

    # Fully Connected Layers
    x = combined
    for nNeurons in denseNeurons:
        x = layers.Dense(nNeurons, activation='relu', kernel_regularizer=reg_dense)(x)

    # Output Layer
    output = layers.Dense(1, activation='linear')(x)

    # Define the Model 
    model = Model(inputs=[input_3d,input_energies], outputs=output)

    #compile the model with no metrics for now
    model.compile(optimizer='adam',
                loss=LossFunction)
    return model

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

"""     DATASET LOADING     """


dataset = tf.data.Dataset.load(TrainDsPath)
test_dataset = tf.data.Dataset.load(TestDsPath)

with h5py.File(hdf5_file, "r") as hdf:
    dataset_size = len(hdf["data"])
    print(f"Train dataset size: {dataset_size}")

train_dataset=dataset.take(int(0.8*dataset_size*p))
train_dataset=train_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_ds2=dataset.skip(int(0.8*dataset_size*p)).take(int(0.2*dataset_size*p))
test_ds2=test_ds2.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

#with h5py.File(test_hdf5_file, "r") as hdf:
#    dataset_size = len(hdf["data"])
#    print(f"Test dataset size: {dataset_size}")
val_dataset=test_dataset.take(int(int(0.3*dataset_size*p)))
val_dataset=val_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
test_ds=test_dataset.skip(int(int(0.3*dataset_size*p))).take(int(int(0.1*dataset_size*p)))
test_ds=test_ds.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print("Dane treningowe, walidacyjne i testowe ready")

"""     MAIN BODY   """

early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=3,                 # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restore model weights from the best epoch
)

# DEFINE THE MODEL
models,names = [],[]

# EXAMPLE MODEL
models.append(GetModel(convFilters=[32,32,16],denseNeurons=[64,32,16],LossFunction=snorm_MSE))
names.append("snorm_MSE")


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
            epochs=20,
            callbacks=[early_stopping,tboard_callback],
            verbose=2
            )
        #Generowanie danych
        y_pred=model.predict(test_ds,verbose=0).flatten()
        y_true = np.array([label.numpy() for _, label in test_ds]).flatten()
        y_pred2=model.predict(test_ds2,verbose=0).flatten()
        y_true2 = np.array([label.numpy() for _, label in test_ds2]).flatten()
        y_true=np.concatenate((y_true,y_true2),axis=0)
        y_pred=np.concatenate((y_pred,y_pred2),axis=0)

    print(f"Saving model {name}")
    RMS, ME, MAE, normMAE = Metrics(y_true, y_pred, Energies)
    SaveModel(ME,MAE,normMAE,RMS,name, model)

    # Saving model history to CSV
    combined_array = np.vstack(list(history.history.values())).T
    header = list(history.history.keys())
    with open(f"{name}_history"+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(combined_array)