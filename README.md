## DINet
DINet is a dual-path deep learning framework for quantitative MIC prediction against H. pylori. It integrates GNN and BiLSTM with channel attention fusion. Interpretability via feature perturbation and graph attention identifies key substructures for drug design.

## Directory Structure

DINet_Project/
├── dtsd_outlier_filter.py  # Outlier filtering script
├── train.py                # Main execution script
├── DINet/                
│   ├── __init__.py
│   ├── data/               # Data handling
│   ├── model/              # Model architecture
│   ├── tool/               # Utilities
│   └── train/              # Training logic
└── README.md     

## Requirements
The code is tested with **Python 3.9** and **PyTorch 2.4.0 (CUDA 12.4)**. 

Key dependencies include:
- `torch` == 2.4.0
- `torch-geometric` == 2.6.1 (PyTorch Geometric)
- `rdkit` == 2024.09.6
- `pandas` == 2.2.2
- `scikit-learn` == 1.6.1
- `optuna` == 4.3.0
- 
You can install the required packages via pip

## Example

###Dataset
The input data should be provided in a .csv format. The file must contain at least two columns: smiles  and mic .

###Data Preprocessing (Outlier Filtering)
Before training, we provide a script to filter potential outliers using a robust cross-validation strategy based on the DTSD (Deviation of Test Set Distribution) metric. Run the script and follow the terminal prompts to input your file paths and hyperparameters:

```shell
python dtsd_outlier_filter.py
```

###Running
Training can be started using the following command. You need to specify the path to your filtered dataset, the save path, the number of epochs, and the batch size.

```shell
python train.py --data_path path/to/filtered_data.csv --save_path model_save/ --epochs  --batch_size 
```
