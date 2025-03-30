# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:40:26 2025

@author: spbki
"""

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
    
    model.save('model+'+'filename')
    

#Kompilacja
metrics=[]
for E in Energies:
    metrics.append(ValLossForE(E=E,loss_func=snorm_RMS))
model.compile(optimizer='adam',
              loss=snorm_RMS,
              metrics=metrics) 


#Callbaki
csv_logger = tf.keras.callbacks.CSVLogger("training_log5.csv", append=True)
logs = "logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logs,
                                                 histogram_freq=1,
                                                 profile_batch='420,430')

with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        shuffle=True,
        epochs=20,                 # Set a high number of epochs; EarlyStopping will stop early
        callbacks=[early_stopping, csv_logger,tboard_callback],  # Add EarlyStopping to callbacks
        verbose=1
    )
print("Wytrenowane!")

#Generowanie danych
y_pred=model.predict(test_ds).flatten()
y_true = np.array([label.numpy() for _, label in test_ds]).flatten()
y_pred2=model.predict(test_ds2).flatten()
y_true2 = np.array([label.numpy() for _, label in test_ds2]).flatten()
y_true=np.concatenate((y_true,y_true2),axis=0)
y_pred=np.concatenate((y_pred,y_pred2),axis=0)


RMS, ME, MAE, normMAE = Metrics(y_true, y_pred, Energies)
SaveModel(ME,MAE,normMAE, RMS,nazwa[ID] )
