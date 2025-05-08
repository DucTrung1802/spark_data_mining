# seminar_4

# 🧠 ML Paper Title Generation

Generate paper titles from ML ArXiv abstracts using transformer models with experiment tracking via MLflow.

## 🚀 Overview

This project trains a sequence-to-sequence transformer model to generate paper titles from ML ArXiv abstracts. It uses PySpark for preprocessing, Hugging Face Transformers for modeling, and MLflow for experiment tracking. The training process leverages modern tools and runs on Python 3.12 with CUDA 12.4 for GPU acceleration.


## ⚙️ Environment

- **Python:** 3.12  
- **CUDA:** 12.4  
- **Frameworks:** PySpark, Hugging Face Transformers, MLflow, PyTorch


## 📦 Setup Instructions

### 1. 📁 Download Data

```bash
mkdir data
cd data
wget https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/resolve/main/ML-Arxiv-Papers.csv
cd ..
```

### 2. 🛠️ Set Up Environment

```bash
conda create -n title python=3.12
conda activate title
```

### 3. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

If using CUDA 12.4, ensure you install PyTorch with the appropriate version:

```bash
pip install torch==2.3.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html
```


## 📊 Track Experiments with MLflow

### Start MLflow UI

```bash
mlflow ui --port 5000
```

Access the MLflow dashboard at: `http:/localhost:5000`


## 🏃‍♂️ Run Training

### Train the Model

```bash
python main.py
```

You can monitor training progress live via the MLflow UI.

### Keep track of training process on MLFlow at 
http://localhost:5000