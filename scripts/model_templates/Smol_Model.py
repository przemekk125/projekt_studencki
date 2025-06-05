# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:40:09 2025

@author: spbki
"""

#Tu zdefiniuj model
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

# Define the Model
model = Model(inputs=[input_3d], outputs=output)