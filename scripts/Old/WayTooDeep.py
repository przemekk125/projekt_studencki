# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:34:58 2025

@author: spbki
"""
import h5py
import tensorflow as tf
import numpy as np
import math
import csv
import time
from tensorflow.keras import layers, models, Model, Input, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib.gridspec import GridSpec

def Load_Classical(File1,File2):
    with open(File1) as csv_file:
        csv_read=csv.reader(csv_file, delimiter=',')
        reg=list(csv_read)
    kTE=[]
    kMAE=[]
    for i in range(1,len(reg)):
        kTE.append(float(reg[i][0][0:5]))
        if reg[i][0][10]=='\t':
            kMAE.append(float(reg[i][0][11:16]))
        else:
            kMAE.append(float(reg[i][0][10:16]))
    kTE=np.array(kTE)
    kMAE=np.array(kMAE)
    with open(File2) as csv_file:
        csv_read=csv.reader(csv_file, delimiter=',')
        reg=list(csv_read)
    kRMS=[]
    kSTD=[]
    for i in range(1,len(reg)):
        if reg[i][0][10]=='\t':
            kRMS.append(float(reg[i][0][11:16]))
        else:
            kRMS.append(float(reg[i][0][10:16]))
        kSTD.append(float(reg[i][0][-8:-1]))
    kRMS=np.array(kRMS)
    kSTD=np.array(kSTD)
    return kTE, kMAE, kRMS, kSTD

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

def test_hdf5_generator(test_hdf5_file):
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
    
def gaussian(x, mu, sigma, amplitude):
    return amplitude * norm.pdf(x, mu, sigma)

def fit_gaussian_to_histogram(dEnergy, E, n_bins=30):
    # Construct the histogram
    hist, bin_edges = np.histogram(dEnergy, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess for the fitting parameters
    initial_sigma = np.std(dEnergy)
    initial_amplitude = np.max(hist)

    # Fix the mean to E
    mu_fixed = E
    # Fit the Gaussian function to the histogram data
    # Only fit sigma and amplitude, keep mu fixed
    popt, pcov = curve_fit(
        lambda x, sigma, amplitude: gaussian(x, mu_fixed, sigma, amplitude),
        bin_centers, hist, p0=[initial_sigma, initial_amplitude]
    )
    # Extract the fitted parameters
    sigma_fit, amplitude_fit = popt

    # Calculate the goodness of fit (R-squared)
    ss_res = np.sum((hist - gaussian(bin_centers, mu_fixed, sigma_fit, amplitude_fit))**2)
    ss_tot = np.sum((hist - np.mean(hist))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return mu_fixed, sigma_fit, r_squared

def Gaussowisko(E_pred, E_true, Energies,n_bins, image_name=None):
    sigmas=[]
    fig, axs = plt.subplots(math.ceil(len(Energies)/3), 3, figsize=(15, 5 * math.ceil(len(Energies)/3)))
    axs = axs.flatten()
    for i in range(0,len(Energies)):
        E=Energies[i]
        mu, sigma, r_squared = fit_gaussian_to_histogram(E_pred[E_true==E], E)
        axs[i].hist(E_pred[E_true==E], bins=n_bins, density=True, alpha=0.6, color='g', label='Histogram')
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        axs[i].plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
        axs[i].text(0.05, 0.95, f'$\mu$ = {mu:.2f}\n$\sigma$ = {sigma:.2f}\nE = {E}\n$R^2$ = {r_squared:.2f}',transform=axs[i].transAxes, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        axs[i].set_title(f'Histogram and Gaussian Fit for E = {E}')
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
        axs[i].legend()
        axs[i].grid(True)
        sigmas.append(sigma)
    plt.tight_layout()
    plt.show()
    if image_name!=None:
        filename = f"{image_name}.png"
        plt.savefig(filename)
    plt.close()
    return np.array(sigmas)

def Metrics(E_true,E_pred,Energies):
    RMS=np.zeros((len(Energies)))
    ME=np.zeros((len(Energies)))
    MAE=np.zeros((len(Energies)))
    for i in range(0,len(Energies)):
        RMS[i]=np.sqrt(np.mean((E_true[E_true==Energies[i]]-E_pred[E_true==Energies[i]])**2))
        ME[i]=np.mean(E_pred[E_true==Energies[i]]-E_true[E_true==Energies[i]])
        MAE[i]=np.mean(np.abs(E_pred[E_true==Energies[i]]-E_true[E_true==Energies[i]]))
    return RMS, ME, MAE, MAE/np.sqrt(Energies)
    
def generate_subplots(Energies, ME, kMAE, MAE, MAE2,  knormMAE, normMAE, normMAE2, kRMS, RMS, RMS2, kSTD, STD, STD2, TE, plot_correction=False, image_name=None, path_to_data=None):
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    if plot_correction:
        axes[0, 1].scatter(Energies, MAE2, label='model corrected with ME', marker='+', color='red')
        axes[1, 0].scatter(Energies, normMAE2, label='model corrected with ME', marker='+', color='red')
        ratio2=[]
        for E in TE:
            ratio2.append(MAE2[Energies==E]/kMAE[TE==E])
        axes[1, 1].scatter(TE, ratio2, label='model corrected with ME', marker='+', color='red')
        axes[2, 0].scatter(Energies, RMS2, label='model corrected with ME', marker='+', color='red')
        axes[2, 1].scatter(Energies, STD2, label='model corrected with ME', marker='+', color='red')
    if path_to_data!=None:
        # Step 1: Initialize empty lists to store the columns
        mME = []
        mMAE = []
        mnormMAE = []
        mRMS = []
        mSTD = []
        mMAE2 = []
        mnormMAE2 = []
        mRMS2 = []
        mSTD2 = []
        
        # Step 2: Read the CSV file
        with open(path_to_data, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                mME.append(float(row['ME']))
                mMAE.append(float(row['MAE']))
                mnormMAE.append(float(row['normMAE']))
                mRMS.append(float(row['RMS']))
                mSTD.append(float(row['STD']))
                mMAE2.append(float(row['MAE2']))
                mnormMAE2.append(float(row['normMAE2']))
                mRMS2.append(float(row['RMS2']))
                mSTD2.append(float(row['STD2']))
        
        axes[0, 0].scatter(Energies, np.array(mME), label='loaded model', marker='o', color='black')
        axes[0, 1].scatter(Energies, np.array(mMAE), label='loaded model', marker='o', color='black')
        axes[1, 0].scatter(Energies, np.array(mnormMAE), label='loaded model', marker='o', color='black')
        axes[2, 0].scatter(Energies, np.array(mRMS),label='loaded model', marker='o', color='black')
        axes[2, 1].scatter(Energies, np.array(mSTD), label='loaded model', marker='o', color='black')
        ratio3=[]
        for E in TE:
            ratio3.append(np.array(mMAE)[Energies==E]/kMAE[TE==E])
        axes[1, 1].scatter(TE, ratio3, label='Ratio (loaded model)/klasyka', marker='+', color='black')
        if plot_correction:
            axes[0, 1].scatter(Energies, np.array(mMAE2), label='corrected loaded model', marker='+', color='black')
            axes[1, 0].scatter(Energies, np.array(mnormMAE2), label='corrected loaded model', marker='+', color='black')
            axes[2, 0].scatter(Energies, np.array(mRMS2),label='corrected loaded model', marker='+', color='black')
            axes[2, 1].scatter(Energies, np.array(mSTD2), label='corrected loaded model', marker='+', color='black')
            ratio3=[]
            for E in TE:
                ratio3.append(np.array(mMAE)[Energies==E]/kMAE[TE==E])
            axes[1, 1].scatter(TE, ratio3, label='Ratio (loaded model)/klasyka', marker='o', color='black')
            ratio4=[]
            for E in TE:
                ratio4.append(np.array(mMAE2)[Energies==E]/kMAE[TE==E])
            axes[1, 1].scatter(TE, ratio4, label='Ratio (corrected loaded model)/klasyka', marker='+', color='black')


    # First row
    axes[0, 0].scatter(Energies, ME, label='model', marker='o', color='red')
    axes[0, 0].set_xlabel('Energies')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].scatter(TE, kMAE, label='klasyka', marker='*', color='gray')
    axes[0, 1].scatter(Energies, MAE, label='model', marker='o', color='red')
    axes[0, 1].set_xlabel('Energies')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Second row
    axes[1, 0].scatter(TE, knormMAE, label='klasyka', marker='*', color='gray')
    axes[1, 0].scatter(Energies, normMAE, label='model', marker='o', color='red')
    axes[1, 0].set_xlabel('Energies')
    axes[1, 0].set_ylabel('Normalized Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    ratio=[]
    for E in TE:
        ratio.append(MAE[Energies==E]/kMAE[TE==E])
    axes[1, 1].scatter(TE, ratio, label='Ratio model/klasyka', marker='o', color='red')
    axes[1, 1].set_xlabel('Energies')
    axes[1, 1].set_ylabel('Ratio MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Third row
    axes[2, 0].scatter(TE, kRMS, label='klasyka', marker='*', color='gray')
    axes[2, 0].scatter(Energies, RMS, label='model', marker='o', color='red')
    axes[2, 0].set_xlabel('Energies')
    axes[2, 0].set_ylabel('RMS')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].scatter(TE, kSTD, label='klasyka', marker='*', color='gray')
    axes[2, 1].scatter(Energies, STD, label='model', marker='o', color='red')
    axes[2, 1].set_xlabel('Energies')
    axes[2, 1].set_ylabel('STD')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    if image_name!=None:
        filename = f"{image_name}.png"
        plt.savefig(filename)
    plt.show()
    plt.close()   

def plot_training_data(Energies, path_to_training_logs, path_to_image, image_name=None):
    # Create a list to store the values for each energy category
    energy_values = [[] for _ in range(len(Energies))]
    val_energy_values = [[] for _ in range(len(Energies))]

    # Open the CSV file
    with open(path_to_training_logs, mode='r') as file:
        csv_reader = csv.DictReader(file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Iterate over each energy value and extract the corresponding column
            for i, energy in enumerate(Energies):
                column_name = f'loss_for_{energy / 10:.1f}_GeV'
                val_name = f'val_loss_for_{energy / 10:.1f}_GeV'
                if column_name in row:
                    energy_values[i].append(float(row[column_name]))
                if val_name in row:
                    val_energy_values[i].append(float(row[val_name]))

    # Create a figure with GridSpec for custom layout
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 1])

    # Create axes for the image spanning the entire first column
    ax_image = fig.add_subplot(gs[:, 0])
    ax_image.imshow(mpimg.imread(path_to_image))
    ax_image.axis('off')  # Hide the axes

    # Create axes for the scatter plots in the second column
    ax_scatter1 = fig.add_subplot(gs[0, 1])
    ax_scatter2 = fig.add_subplot(gs[1, 1])

    # Normalize energies for color mapping
    norm = plt.Normalize(vmin=min(Energies), vmax=max(Energies))
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])

    # Plot scatter plots in the second column
    for i, energy in enumerate(Energies):
        x = range(len(energy_values[i]))
        y = energy_values[i]
        ax_scatter1.scatter(x, y, color=sm.to_rgba(energy),s=10, label=f'{energy / 10:.1f} GeV')

        x_val = range(len(val_energy_values[i]))
        y_val = val_energy_values[i]
        ax_scatter1.scatter(x_val, y_val, color=sm.to_rgba(energy), marker='x',s=100, label=f'{energy / 10:.1f} GeV')
    

    # Add color bar
    #cbar = fig.colorbar(sm, ax=[ax_scatter1, ax_scatter2], orientation='vertical', pad=0.4)
    #cbar.set_label('Energy (GeV)')

    # Set titles and labels
    ax_scatter1.set_xlim(left=0.5)
    ax_scatter1.set_ylim(bottom=3,top=1.2*np.max(energy_values[-1][1:-1]))
    ax_scatter1.set_title('Loss vs Epoch by energy level')
    ax_scatter1.set_xlabel('Epoch')
    ax_scatter1.set_ylabel('Loss')

    for i, energy in enumerate(Energies):
        x = range(len(energy_values[i]))
        y = energy_values[i]
        y=np.array(y)/(y[1]+0.001)
        ax_scatter2.scatter(x, y, color=sm.to_rgba(energy),s=10, label=f'{energy / 10:.1f} GeV')

        x_val = range(len(val_energy_values[i]))
        y_val = val_energy_values[i]
        y_val=np.array(y_val)/(y_val[1]+0.001)
        ax_scatter2.scatter(x_val, y_val, color=sm.to_rgba(energy), marker='x',s=100, label=f'{energy / 10:.1f} GeV')
    

    # Add color bar
    #cbar = fig.colorbar(sm, ax=[ax_scatter1, ax_scatter2], orientation='vertical', pad=0.4)
    #cbar.set_label('Energy (GeV)')

    # Set titles and labels
    ax_scatter2.set_xlim(left=0.5)
    ax_scatter2.set_ylim(bottom=0.95)
    ax_scatter2.set_title('Normalized loss vs Epoch by energy level')
    ax_scatter2.set_xlabel('Epoch')
    ax_scatter2.set_ylabel('Normalized loss')
    

    # Adjust layout
    plt.tight_layout()
    plt.show()
    if image_name!=None:
        filename = f"{image_name}.png"
        plt.savefig(filename)
    plt.close()    

def Save_model_metrics(ME, MAE, normMAE, RMS, STD, MAE2, normMAE2, RMS2, STD2, filename):
    combined_array = np.vstack((ME, MAE, normMAE, RMS, STD, MAE2, normMAE2, RMS2, STD2)).T

    # Step 2: Define the header for the CSV file
    header = ['ME', 'MAE', 'normMAE', 'RMS', 'STD', 'MAE2', 'normMAE2', 'RMS2', 'STD2']
    
    # Step 3: Write the combined array to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(combined_array)

def norm_MAE(y_true, y_pred):   
    """Normalized Mean Absolute Error (MAE) divided by y_true."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.abs(y_true - y_pred)
    norm_mae = mae / tf.maximum(y_true, 1e-10)  # Avoid division by zero
    return tf.reduce_mean(norm_mae, axis=-1)

def snorm_MAE(y_true, y_pred):
    """Square Root Normalized Mean Absolute Error (MAE) divided by sqrt(y_true)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mae = tf.abs(y_true - y_pred)
    snorm_mae = mae / tf.maximum(tf.sqrt(y_true), 1e-10)  # Avoid division by zero
    return tf.reduce_mean(snorm_mae, axis=-1)

def norm_MSE(y_true, y_pred):
    """Normalized Root Mean Square (RMS) divided by y_true."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    norm_rms = rms / tf.maximum(tf.square(y_true), 1e-10)  # Avoid division by zero
    return tf.reduce_mean(norm_rms, axis=-1)

def snorm_MSE(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = rms / tf.maximum(y_true , 1e-10)  # Avoid division by zero
    return tf.reduce_mean(snorm_rms, axis=-1)

def snorm_RMS(y_true, y_pred):
    """Square Root Normalized Root Mean Square (RMS) divided by sqrt(y_true)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = rms / tf.maximum(y_true, 1e-10)  # Avoid division by zero
    return tf.sqrt(tf.reduce_mean(snorm_rms, axis=-1))

def norm_RMS(y_true, y_pred):
    """Square Root Normalized Root Mean Square (RMS) divided by y_true."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rms = tf.square(y_true - y_pred)
    snorm_rms = rms / tf.maximum(tf.square(y_true), 1e-10)  # Avoid division by zero
    return tf.sqrt(tf.reduce_mean(snorm_rms, axis=-1))

def norm_Huber(d):
    """Normalized Huber loss divided by y_true."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.abs(y_true - y_pred)
        is_small_error = error < d
        squared_loss = tf.square(error) / 2
        linear_loss = d * (error - d / 2)
        huber_loss = tf.where(is_small_error, squared_loss, linear_loss)
        norm_huber_loss = huber_loss / tf.maximum(y_true, 1e-10)  # Avoid division by zero
        return tf.reduce_mean(norm_huber_loss, axis=-1)
    return loss

def snorm_Huber(d):
    """Square Root Normalized Huber loss divided by sqrt(y_true)."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.abs(y_true - y_pred)
        is_small_error = error < d
        squared_loss = tf.square(error) / 2
        linear_loss = d * (error - d / 2)
        huber_loss = tf.where(is_small_error, squared_loss, linear_loss)
        snorm_huber_loss = huber_loss / tf.maximum(tf.sqrt(y_true), 1e-10)  # Avoid division by zero
        return tf.reduce_mean(snorm_huber_loss, axis=-1)
    return loss

    
