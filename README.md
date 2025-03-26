# GraphCheck: Multi-Path Fact-Checking with Entity-Relationship Graphs

GraphCheck is a fact-checking framework that leverages entity-relation graphs to perform multi-path reasoning for claim verification. This repository includes code, data, and instructions to reproduce the results from our paper, [GraphCheck: Multi-Path Fact-Checking with Entity-Relationship Graphs (Jeon et al., 2025)](https://arxiv.org/abs/2502.20785)

This repository contains the source code for .

## **Setup Virtual Environment**

To set up the required environment, use the following commands:

```bash
conda create --name graphcheck python=3.10
conda activate graphcheck
pip install -r requirements.txt
```

---

## **Experiments on HOVER (Jiang et al., 2020)**

### **Build BM25 Index**

Before running experiments, prepare the BM25 index using:

```bash
sh prepare_hover.sh
```

### **Running Direct**

To run Direct on the **dev set**:

```bash
python direct.py \
    --dataset hover \
    --input_filename dev.json
```

### **Running GraphCheck**

To run **GraphCheck** on the **dev set**:

```bash
python graphcheck.py \
    --dataset hover \
    --input_filename dev.json
```

#### **Note:**

We provide the pre-constructed graph list in the `results` folder, so you do **not** need to rebuild the graph. However, if you wish to construct graphs from scratch, use the following command **with your OpenAI API key**:

```bash
python graphcheck.py \
    --dataset hover \
    --input_filename dev.json \
    --force_new_construction \
    --openai_api_key YOUR_OPENAI_API_KEY
```

Note that using the OpenAI API incurs costs. We also provide an option to use the **OpenAI Batch API**, which can reduce costs by **50%**, though it may slow down graph construction.

```bash
python graphcheck.py \
    --dataset hover \
    --input_filename dev.json \
    --force_new_construction \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --use_openai_batch_api
```

### **Running DP-GraphCheck**

Running **DP-GraphCheck** follows the same procedure as **GraphCheck**:

```bash
python dp_graphcheck.py \
    --dataset hover \
    --input_filename dev.json
```

For graph reconstruction:

```bash
python dp_graphcheck.py \
    --dataset hover \
    --input_filename dev.json \
    --force_new_construction \
    --openai_api_key YOUR_OPENAI_API_KEY
```

For cost-efficient graph reconstruction:

```bash
python dp_graphcheck.py \
    --dataset hover \
    --input_filename dev.json \
    --force_new_construction \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --use_openai_batch_api
```

---

## **Experiments on EX-FEVER (Aly et al., 2021)**

Similar steps apply for **EX-FEVER** dataset.

#### **Note:**

As mentioned in our paper, we provide a modified version of the **EX-FEVER dataset** in `datasets/ex-fever/claims/`, where the original **Not Enough Information (NEI)** label has been removed, leaving only `SUPPORTED` or `NOT SUPPORTED` labels.

### **Build BM25 Index**

```bash
sh prepare_ex-fever.sh
```

### **Running Experiments on Test Set**

To run **Direct**:

```bash
python direct.py \
    --dataset ex-fever \
    --input_filename test_nei_x.json
```

To run **GraphCheck**:

```bash
python graphcheck.py \
    --dataset ex-fever \
    --input_filename test_nei_x.json
```

To run **DP-GraphCheck**:

```bash
python dp_graphcheck.py \
    --dataset ex-fever \
    --input_filename test_nei_x.json
```
