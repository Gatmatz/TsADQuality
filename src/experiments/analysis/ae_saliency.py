"""
Explainability (Saliency Maps) for Autoencoder

This script demonstrates how data quality issues (like noise or adversarial attacks)
shift the "attention" of the Autoencoder, leading to failures in detecting true anomalies.

It uses Gradient Saliency (the derivative of the reconstruction loss with respect to the input)
to visualize which data points influenced the model's error the most. High saliency (dark red points)
indicates that the model focused its attention on those points to compute the anomaly score.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Setup project paths
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import load_tsb_file
from src.ts_corruptor.core import TSCorruptor
from src.ts_corruptor.injectors import inject_white_noise_snr, inject_spikes

def get_ae_saliency_map(model, input_window):
    """
    Υπολογίζει το Saliency Map (Gradients του MSE Loss ως προς την Είσοδο)
    """
    # Το Keras AE μοντέλο περιμένει shape (batch_size, window_size, features)
    input_tensor = tf.convert_to_tensor(input_window.reshape(1, -1, 1), dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        # Ανακατασκευή (Prediction)
        reconstruction = model(input_tensor)
        # Σφάλμα ανακατασκευής
        loss = tf.reduce_mean(tf.square(input_tensor - reconstruction))
        
    # Πόσο επηρεάζει η κάθε τιμή εισόδου το συνολικό σφάλμα;
    gradients = tape.gradient(loss, input_tensor)
    
    # Παίρνουμε το απόλυτο μέγεθος (magnitude)
    saliency = tf.abs(gradients).numpy().flatten()
    
    # Min-Max Normalization (0 έως 1) για καλύτερο visualization
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency) + 1e-8)
    return saliency, reconstruction.numpy().flatten()

def main():
    # Βρίσκουμε ένα pre-trained μοντέλο (π.χ. από το YAHOO benchmark)
    target_filename = 'Yahoo_A1real_1_data.out'
    pretrained_dir = os.path.join(project_root, 'results', 'pretrained_models', 'ae')
    
    model_dir = os.path.join(pretrained_dir, target_filename)
    if not os.path.exists(model_dir):
        # Fallback: Παίρνουμε το πρώτο διαθέσιμο μοντέλο
        available_models = [d for d in os.listdir(pretrained_dir) if os.path.isdir(os.path.join(pretrained_dir, d))]
        if available_models:
            target_filename = available_models[0]
            model_dir = os.path.join(pretrained_dir, target_filename)
        else:
            print("No pretrained models found!")
            return

    print(f"Selected Dataset/Model: {target_filename}")
    
    # Φόρτωση του pre-trained Keras Model
    model_path = os.path.join(model_dir, 'model.keras')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return

    ae_model = tf.keras.models.load_model(model_path)
    window_size = ae_model.input_shape[1]
    print(f"Model loaded successfully. Expected Window Size: {window_size}")
    
    # Αναζήτηση του dataset file (χρησιμοποιούμε τη ρουτίνα φόρτωσης που έχεις ήδη)
    subset_csv = os.path.join(project_root, "results", "tables", "robust_subset_TSB.csv")
    file_path = None
    if os.path.exists(subset_csv):
        df_subset = pd.read_csv(subset_csv)
        file_row = df_subset[df_subset['baseline_name'] == target_filename]
        if not file_row.empty:
            file_path_rel = file_row.iloc[0]['filepath']
            file_path = os.path.join(project_root, 'TSB-UAD', file_path_rel)
            
    if file_path is None or not os.path.exists(file_path):
        # Dummy fallback filepath
        file_path = os.path.join(project_root, 'TSB-UAD', 'data', 'benchmark', 'YAHOO', target_filename)
    
    if not os.path.exists(file_path):
        print(f"Dataset file not found at {file_path}. Please update target_filename.")
        return

    data, label, _ = load_tsb_file(file_path)
    
    # Preprocessing (ίδιο με την εκπαίδευση)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    print(f"Data Length: {len(scaled_data)}, Anomalies: {np.sum(label)}")

    # Βρίσκουμε ένα παράθυρο που να περιέχει την ανωμαλία στο κέντρο του
    anomaly_indices = np.where(label == 1)[0]
    
    if len(anomaly_indices) > 0:
        # Επιλέγουμε την πρώτη ανωμαλία (στη μέση περίπου των anomalies για σιγουριά)
        center = anomaly_indices[len(anomaly_indices)//2]
        start_idx = max(0, center - window_size // 2)
        end_idx = start_idx + window_size
        
        # Αν βγούμε εκτός ορίων, προσαρμόζουμε
        if end_idx > len(scaled_data):
            end_idx = len(scaled_data)
            start_idx = end_idx - window_size
            
        clean_window = scaled_data[start_idx:end_idx]
        window_labels = label[start_idx:end_idx]
        
        # Εισαγωγή Σφαλμάτων (Corruptions)
        # Δοκιμάζουμε ένα White Noise Injector για να δούμε πώς αλλάζει το Saliency
        temp_df = pd.DataFrame({'value': clean_window, 'is_anomaly': window_labels})
        corruptor = TSCorruptor(temp_df, value_col='value', label_col='is_anomaly', corruption_target='global')
        inject_white_noise_snr(corruptor, snr_db=5.0)
        corrupted_window = corruptor.get_corrupted_df()['value'].to_numpy()
        
        # Υπολογισμός Saliency
        saliency_clean, recon_clean = get_ae_saliency_map(ae_model, clean_window)
        saliency_corrupt, recon_corrupt = get_ae_saliency_map(ae_model, corrupted_window)
    
        # Ζωγραφίζουμε το αποτέλεσμα!
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
        
        # Γράφημα 1: Καθαρά Δεδομένα
        axes[0].plot(clean_window, label='Clean Input', color='#1f77b4', linewidth=1.5)
        # Χρωματίζουμε την πραγματική ανωμαλία στο background (με ανοιχτό κόκκινο)
        for i, l in enumerate(window_labels):
            if l == 1:
                axes[0].axvspan(i-0.5, i+0.5, color='red', alpha=0.2, lw=0)
                axes[1].axvspan(i-0.5, i+0.5, color='red', alpha=0.2, lw=0)
                
        scatter_clean = axes[0].scatter(range(len(clean_window)), clean_window, 
                        c=saliency_clean, cmap='Reds', s=saliency_clean*200, zorder=5, edgecolor='black', linewidths=0.5)
        axes[0].set_title('Autoencoder Attention on Clean Data (Shaded Red = True Anomaly)', fontsize=14)
        axes[0].legend(loc='upper left')
        plt.colorbar(scatter_clean, ax=axes[0], label='Saliency (Attention)')
        
        # Γράφημα 2: Corrupted Δεδομένα
        axes[1].plot(corrupted_window, label='Corrupted Input (Noise)', color='#ff7f0e', linewidth=1.5)
        scatter_corrupt = axes[1].scatter(range(len(corrupted_window)), corrupted_window, 
                        c=saliency_corrupt, cmap='Reds', s=saliency_corrupt*200, zorder=5, edgecolor='black', linewidths=0.5)
        axes[1].set_title('Autoencoder Attention Shifted by Noise!', fontsize=14)
        axes[1].legend(loc='upper left')
        plt.colorbar(scatter_corrupt, ax=axes[1], label='Saliency (Attention)')
        
        plt.tight_layout()
        output_path = os.path.join(project_root, 'results', 'experiments', 'analysis')
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, 'ae_explainability_saliency.png')
        
        plt.savefig(save_path, dpi=300)
        print(f"Saliency plot saved as '{save_path}'")
        plt.show()
    else:
        print("Δεν βρέθηκαν ανωμαλίες σε αυτή τη χρονοσειρά!")

if __name__ == "__main__":
    main()
