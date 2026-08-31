"""
Shared data loading utilities for the thesis project.

Handles NASA-style train+test concatenation (SMAP, MSL, KPI datasets)
so that baseline and corruption experiments use identical data.
"""
import os
import numpy as np
import pandas as pd


def remap_filepath(filepath, project_root):
    """Remap an absolute filepath from the CSV to the current machine's project root.

    The subset CSV stores absolute Windows paths (e.g. C:\\Users\\gkost\\thesis_timeseries\\...).
    When running on a different machine (e.g. Linux cluster), we extract the relative
    portion after 'thesis_timeseries' and rejoin it with the actual project_root.

    If the file already exists at the given path, it is returned unchanged.
    """
    if os.path.exists(filepath):
        return filepath
    for sep in ['\\', '/']:
        marker = f'thesis_timeseries{sep}'
        idx = filepath.find(marker)
        if idx >= 0:
            rel = filepath[idx + len(marker):]
            rel = rel.replace('\\', '/').replace('/', os.sep)
            remapped = os.path.join(project_root, rel)
            return remapped
    return filepath


def load_tsb_file(filepath):
    """Load a TSB-UAD time series file, handling NASA train+test concatenation.

    For files where both .test. and .train. versions exist (NASA SMAP/MSL/KPI),
    the two splits are concatenated (train first, then test) following the
    TSB-UAD benchmark convention for unsupervised methods.

    Parameters
    ----------
    filepath : str
        Path to the .test. or standalone .out file.

    Returns
    -------
    data : np.ndarray (float64)
        Time series values.
    label : np.ndarray (int)
        Binary anomaly labels.
    canonical_name : str
        Canonical filename for result matching.  Uses the .test. name
        as-is (same as the subset CSV) so that all experiments share
        a single key.
    """
    filename = os.path.basename(filepath)
    canonical_name = filename  # always matches subset CSV

    if '.test.' in filename:
        train_path = filepath.replace('.test.', '.train.')
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path, header=None)
            df_test = pd.read_csv(filepath, header=None)
            data = np.concatenate([df_train.iloc[:, 0].values,
                                   df_test.iloc[:, 0].values]).astype(float)
            label = np.concatenate([df_train.iloc[:, 1].values,
                                    df_test.iloc[:, 1].values]).astype(int)
        else:
            df = pd.read_csv(filepath, header=None)
            data = df.iloc[:, 0].to_numpy(float)
            label = df.iloc[:, 1].to_numpy(int)
    else:
        df = pd.read_csv(filepath, header=None)
        data = df.iloc[:, 0].to_numpy(float)
        label = df.iloc[:, 1].to_numpy(int)

    # Drop NaNs
    mask = ~np.isnan(data)
    data = data[mask]
    label = label[mask]

    return data, label, canonical_name


def load_tsb_dataframe(filepath):
    """Load a TSB-UAD file as a DataFrame with 'value' and 'is_anomaly' columns.

    Convenience wrapper around :func:`load_tsb_file` for use with
    TSCorruptor-based corruption experiments.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``value`` (float), ``is_anomaly`` (int).
    canonical_name : str
        Canonical filename for result matching.
    """
    data, label, canonical_name = load_tsb_file(filepath)
    df = pd.DataFrame({'value': data, 'is_anomaly': label})
    return df, canonical_name

# This will be for the tsb-ad adaption

def load_tsb_file_ad(filepath):
    """ load a tsb-ad file """
    filename = os.path.basename(filepath)
    # Προσοχή: Τα αρχεία του TSB-AD ΕΧΟΥΝ header (π.χ. 'value', 'Label')!
    # Γι' αυτό ΔΕΝ βάζουμε header=None όπως στο TSB-UAD.
    df = pd.read_csv(filepath).dropna()
    # Το TSB-AD παίρνει όλες τις στήλες εκτός από την τελευταία (0:-1) για data
    data = df.iloc[:, 0:-1].values.astype(float)
    # Η τελευταία στήλη λέγεται 'Label'
    label = df['Label'].astype(int).to_numpy()
    return data, label, filename

def load_tsb_dataframe_ad(filepath):
    """Load a TSB-AD file as a DataFrame with 'value' and 'is_anomaly' columns.

    Convenience wrapper around :func:`load_tsb_file_ad` for use with
    TSCorruptor-based corruption experiments.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``value`` (float), ``is_anomaly`` (int).
    canonical_name : str
        Canonical filename for result matching.
    """
    data, label, canonical_name = load_tsb_file_ad(filepath)
    # Χρησιμοποιούμε ravel() γιατί το data είναι 2D (π.χ. 1000x1) και η στήλη θέλει 1D
    df = pd.DataFrame({'value': data.ravel(), 'is_anomaly': label})
    return df, canonical_name



def load_pretrained_ae(canonical_name, project_root):
    """Load a pre-trained AE model for scoring (no retraining needed).

    Parameters
    ----------
    canonical_name : str
        The filename (e.g. 'S1-ADL4.test.csv@52.out') matching the subset CSV.
    project_root : str
        Project root directory.

    Returns
    -------
    clf : AE_MLP2
        AE model with trained weights loaded. Call clf.predict(scaled_data).
    metadata : dict
        Training metadata (sliding_window, train_ratio, etc.).
    """
    import json
    from tensorflow.keras.models import load_model
    from TSB_UAD.models.AE_2 import AE_MLP2

    pretrained_dir = os.path.join(project_root, "results", "pretrained_models", "ae")
    model_path = os.path.join(pretrained_dir, canonical_name, "model.keras")
    meta_path = os.path.join(pretrained_dir, canonical_name, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained AE model not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"AE metadata not found: {meta_path}")

    with open(meta_path) as f:
        metadata = json.load(f)

    clf = AE_MLP2(slidingWindow=metadata['sliding_window'], epochs=0, verbose=0)
    clf.model_ = load_model(model_path)

    return clf, metadata
