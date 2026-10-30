# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")  


# --------------------------------------------------------
def ask(prompt, cast=str):
    """Prompt the user for input and cast it to the given type."""
    raw = input(f"{prompt}: ").strip()
    return cast(raw)


input_csv_path = ask("Enter input CSV path", str)
output_csv_path = ask("Enter filtered output CSV path", str)
outliers_csv_path = ask("Enter outliers output CSV path", str)

K_FOLDS = ask("Enter K_FOLDS (number of folds)", int)
ITERATIONS = ask("Enter ITERATIONS (number of iterations)", int)
MAD_MULTIPLIER = ask("Enter MAD_MULTIPLIER (MAD multiplier)", float)

ECFP_RADIUS = ask("Enter ECFP_RADIUS (radius)", int)
ECFP_BITS = ask("Enter ECFP_BITS (number of bits)", int)

RF_N_ESTIMATORS = ask("Enter RF_N_ESTIMATORS (number of trees)", int)
RF_MAX_DEPTH = ask("Enter RF_MAX_DEPTH (maximum depth)", int)
# --------------------------------------------------------
# --------------------------------------------------------

print("...")
df = pd.read_csv(input_csv_path)

smiles_col = 'smiles' 
mic_col = 'mic'        

if smiles_col not in df.columns or mic_col not in df.columns:
    raise ValueError(f" {list(df.columns)}")

df = df[[smiles_col, mic_col]].copy()
df.columns = ['smiles', 'mic']

def smiles_to_ecfp(smiles, radius=ECFP_RADIUS, n_bits=ECFP_BITS):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros((1, n_bits), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.flatten()
    except Exception as e:
        return None

fingerprints = []
valid_indices = []

for idx, row in df.iterrows():
    fp = smiles_to_ecfp(row['smiles'])
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)
    else:
        print( )

X = np.array(fingerprints)
y = df.loc[valid_indices, 'mic'].values
original_df = df.loc[valid_indices].reset_index(drop=True)

def filter_outliers_kfold_paper_method(X, y, original_df, k_folds=K_FOLDS, iterations=ITERATIONS, mad_multiplier=MAD_MULTIPLIER):
    n_samples = X.shape[0]
    all_predictions = [[] for _ in range(n_samples)] 

    for iteration in range(iterations):
        print(f"Running iteration {iteration+1}/{iterations}")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42 + iteration)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            for i, pred in zip(test_idx, y_pred):
                all_predictions[i].append(pred)

 
    dtstd_array = np.array([
        np.std(preds, ddof=1) if len(preds) >= 2 else 0.0
        for preds in all_predictions
    ])

    M = np.median(dtstd_array)
    MAD = 1.4826 * np.median(np.abs(dtstd_array - M))
    threshold = M + mad_multiplier * MAD
    print(f"Final threshold: M={M:.4f}, MAD={MAD:.4f}, Threshold={threshold:.4f}")

    reliable_mask = dtstd_array <= threshold
    filtered_df = original_df[reliable_mask].copy()
    outliers_df = original_df[~reliable_mask].copy()
    filtered_df['DTSD'] = dtstd_array[reliable_mask]
    outliers_df['DTSD'] = dtstd_array[~reliable_mask]

    return filtered_df, outliers_df

print("")
filtered_data, outlier_data = filter_outliers_kfold_paper_method(X, y, original_df)

filtered_data.to_csv(output_csv_path, index=False)
outlier_data.to_csv(outliers_csv_path, index=False)

