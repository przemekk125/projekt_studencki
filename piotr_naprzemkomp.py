import h5py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras import layers, models, Model, Input, regularizers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.layers import Lambda # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from datetime import datetime

"""     GLOBALS     """
hdf5_file = "energy_data.h5"
test_hdf5_file = "angle_data.h5"

p=1
Energies=np.sort(np.array([130,140,40,30,200,100,90,170,120,150,190,70,160,60,80,180,110,50,25,75,125]))

""" FUNCTIONS       """
# Define the generator function
def hdf5_generator(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        data = f['data']
        labels = f['labels']
        sum_energies = f['energy']
        num_samples = data.shape[0]

        for i in range(num_samples):
            image = data[i]
            label = labels[i].astype(np.float32).reshape(1) # casting for safety
            sum_energy = sum_energies[i].astype(np.float32).reshape(1) # casting for safety
            image = np.expand_dims(image, axis=-1)

            yield (image, sum_energy), label

def snorm_RMS(y_true, y_pred):
    """Square Root Normalized Root Mean Square (RMS) divided by sqrt(y_true)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = rms / tf.maximum(tf.sqrt(y_true), 1e-10)  # Avoid division by zero
    return tf.sqrt(tf.reduce_mean(snorm_rms, axis=-1))

def GetModel(convFilters:np.array,denseNeurons:np.array):
    # Input Layer
    input_3d = Input(shape=(20, 110, 11, 1,), name='3D_Input')
    input_energies = Input(shape=(1,), name='Energies_Input')
    
    # 3D CNN Path
    x = input_3d
    for nFeatures in convFilters:
        x = layers.Conv3D(nFeatures, kernel_size=(3, 3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Flatten 3D features
    x = layers.Flatten()(x)

    combined = layers.Concatenate(axis=-1)([x, input_energies])  # Concatenate along last axis

    # Fully Connected Layers
    x = combined
    for nNeurons in denseNeurons:
        x = layers.Dense(nNeurons, activation='relu')(x)

    # Output Layer
    output = layers.Dense(1, activation='linear')(x)

    # Define the Model Important to have two imputs
    model = Model(inputs=[input_3d,input_energies], outputs=output)

    #compile the model with no metrics for now
    model.compile(optimizer='adam',
                loss=snorm_RMS)
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

dataset = tf.data.Dataset.from_generator(
    hdf5_generator,
    args=[hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        (tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32, name='x'),
        tf.TensorSpec(shape=(1,), dtype=tf.float32,name = 'energy')),
        tf.TensorSpec(shape=(1,), dtype=tf.int16,name = 'y')
    )
)


print("Train data loaded")

# Define the dataset pipeline
test_dataset = tf.data.Dataset.from_generator(
    hdf5_generator,
    args=[test_hdf5_file],  # Provide the path to your HDF5 file here
    output_signature=(
        (tf.TensorSpec(shape=(20, 110, 11, 1), dtype=tf.float32,name="x"),  # Shape with added color channel
        tf.TensorSpec(shape=(1,), dtype=tf.float32,name = 'energy')),  
        tf.TensorSpec(shape=(1,), dtype=tf.int16,name = 'y')# Label shape
    )
)
print("Validation and test data loaded")
with h5py.File(hdf5_file, "r") as hdf:
    dataset_size = len(hdf["data"])  # Or hdf["labels"], if they have the same length
    print(f"Dataset size: {dataset_size}")

train_dataset=dataset.take(int(0.8*dataset_size*p))
train_dataset=train_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

val_dataset=test_dataset.take(int(int(0.3*dataset_size*p)))
val_dataset=val_dataset.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds=test_dataset.skip(int(int(0.3*dataset_size*p))).take(int(int(0.1*dataset_size*p)))
test_ds=test_ds.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

test_ds2=dataset.skip(int(0.8*dataset_size*p)).take(int(0.2*dataset_size*p))
test_ds2=test_ds2.batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print("Dane treningowe, walidacyjne i testowe ready")

"""     MAIN BODY   """

early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor (e.g., 'val_loss' or 'val_accuracy')
    patience=3,                 # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restore model weights from the best epoch
)

#Tu zdefiniuj modele
models,names = [],[]
models.append(GetModel(convFilters=[32,32,16],denseNeurons=[64,32,16]))
models.append(GetModel(convFilters=[16,8,8],denseNeurons=[32,16,8]))
names = ["BIG","SMALL"]

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
            verbose=0
            )
        #Generowanie danych
        y_pred=model.predict(test_ds).flatten()
        y_true = np.array([label.numpy() for _, label in test_ds]).flatten()
        y_pred2=model.predict(test_ds2).flatten()
        y_true2 = np.array([label.numpy() for _, label in test_ds2]).flatten()
        y_true=np.concatenate((y_true,y_true2),axis=0)
        y_pred=np.concatenate((y_pred,y_pred2),axis=0)

    RMS, ME, MAE, normMAE = Metrics(y_true, y_pred, Energies)
    SaveModel(ME,MAE,normMAE,RMS,name, model)