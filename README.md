# H?c máy và Khai thác d? li?u nâng cao

# 1. Thành viên nhóm và phân công công vi?c

| H? và tên               | MSSV      | Phân công công vi?c                                                 |
|-------------------------|-----------|---------------------------------------------------------------------|
| Hà Thanh Hýõng          | 23007944  | - T?u h?<br>- T?u h?<br>- T?u h? |
| Nguy?n Th? Minh Phý?ng  | 23007937  | - T?u h?<br>- T?u h?<br>- T?u h? |
| L? Ð?c Trung            | 23007933  | - T?u h?<br>- T?u h?<br>- T?u h? |
| Nguy?n Th? Ng?c Uyên    | 23007930  | - T?u h?<br>- T?u h?<br>- T?u h? |


# 2. Hý?ng d?n v? cách t? ch?c và th?c nghi?m chýõng tr?nh

## 2.1. Yêu c?u v? ph?n c?ng

### 2.1.1. CPU (cho Spark và x? l? d? li?u)
- S? l?i: T?i thi?u 4 l?i v?t l? (nên có 8 lu?ng tr? lên)
- T?c ð? xung nh?p: 2.5 GHz ho?c nhanh hõn

### 2.1.2. RAM
- T?i thi?u: 16 GB (Spark và vi?c token hóa T5 c?n nhi?u b? nh?)
- Khuy?n ngh?: 32 GB ð? x? l? mý?t mà hõn

### 2.1.3. GPU (cho vi?c hu?n luy?n và sinh tiêu ð? b?ng mô h?nh T5)

**T?i thi?u:**
- GPU: NVIDIA v?i ít nh?t 8 GB VRAM (ví d?: NVIDIA RTX 3060, Quadro RTX 4000, ho?c Tesla T4)
- CUDA Compute Capability: T? 7.0 tr? lên

**Khuy?n ngh?:**
- GPU: T? 12 GB VRAM tr? lên (ví d?: RTX 3080, A6000, ho?c týõng ðýõng)


### 2.1.4. B? nh? lýu tr?
- SSD v?i ít nh?t **50 GB** dung lý?ng tr?ng

## 2.2. Yêu c?u v? h? ði?u hành

- Linux: Ubuntu 20.04 tr? lên (ýu tiên do h? tr? CUDA t?t) ()
- Windows 10+

## 2.3. Yêu c?u v? ph?n m?m

### 2.3.1. Ph?n m?m

- CUDA: 12.4 (kh?p v?i Torch 2.7.0)
- cuDNN: Phù h?p v?i CUDA 12.4
- Driver GPU: NVIDIA Driver phiên b?n >= 550.x


### 2.3.2. Các thý vi?n Python

- Python 3.12
- pyspark 3.5.5
- transformers 4.51.3
- torch 2.7.0
- mlflow 2.22.0
- sacrebleu 2.5.1
- rouge 1.0.1
- accelerate 1.6.0
- hf_xet 1.1.0

# 3. Hý?ng d?n cài ð?t

**`Hý?ng d?n cài ð?t s? cho h? ði?u hành Ubuntu 20.04`**

## 3.1. Ð?m b?o yêu c?u v? ph?n c?ng và h? ði?u hành t?i bý?c này


## 3.2. C?p nh?t cho h? ði?u hành

M? Terminal ch?y d?ng code sau

```bash
sudo apt update && sudo apt upgrade
```


---

### 1. ?? Download Data

```bash
mkdir data
cd data
wget https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/resolve/main/ML-Arxiv-Papers.csv
cd ..
```

### 2. ??? Set Up Environment

```bash
conda create -n title python=3.12
conda activate title
```

### 3. ?? Install Dependencies

```bash
pip install -r requirements.txt
```

If using CUDA 12.4, ensure you install PyTorch with the appropriate version:

```bash
pip install torch 2.3.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html
```


## ?? Track Experiments with MLflow

### Start MLflow UI

```bash
mlflow ui --port 5000
```

Access the MLflow dashboard at: `http:/localhost:5000`


## ????? Run Training

### Train the Model

```bash
python main.py
```

You can monitor training progress live via the MLflow UI.

### Keep track of training process on MLFlow at 
http://localhost:5000