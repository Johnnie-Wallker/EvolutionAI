# Vision-Language Models for Document Information Extraction
### A UCL & EvolutionAI Collaboration Project

This repository is the official codebase for the UCL project in collaboration with EvolutionAI. It contains a suite of Python scripts for running inference and fine-tuning various state-of-the-art vision-language models (VLMs) on a range of document understanding and visual question answering datasets.

The primary goal is to evaluate and enhance the capabilities of models like **Qwen-VL**, **InternVL**, **Phi-4**, and **GPT-4o** on tasks such as information extraction from receipts, documents, and forms.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Running Inference](#running-inference)
- [Fine-Tuning Models](#fine-tuning-models)
- [Available Experiments](#available-experiments)
- [Utility Scripts](#utility-scripts)

## 🚀 Project Overview

This project provides a framework to:
-   **Run Inference**: Evaluate pre-trained VLMs on several benchmark datasets for document information extraction.
-   **Fine-Tune**: Adapt the Qwen-VL model using Low-Rank Adaptation (LoRA) to improve performance on specific datasets.
-   **Modular Design**: Each experiment is self-contained in its own script, making it easy to run and modify. Utility functions are centralized for reuse.

### Supported Models
-   Qwen2.5-VL-3B-Instruct
-   InternVL3-2B-Instruct
-   Phi-4-multimodal-instruct
-   GPT-4o

### Datasets & Experiments
-   **CORD**: Information extraction from receipts.
-   **DocVQA**: Visual question answering on document images.
-   **SROIE**: Scanned receipt OCR and information extraction.
-   **KLC (kleister-charity)**: Extraction from UK charity annual reports.
-   **Custom Datasets**: Scripts for three different custom datasets (`dataset1`, `dataset2`, `dataset3`) are also included.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Johnnie-Wallker/EvolutionAI.git](https://github.com/Johnnie-Wallker/EvolutionAI.git)
    cd EvolutionAI
    ```

2.  **Install Dependencies:**
    You will need PyTorch with CUDA support. Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for instructions tailored to your system. After installing PyTorch, install the other packages using the appropriate requirements file.

    **For all models EXCEPT `Phi-4-multimodal-instruct`:**
    ```bash
    pip install -r requirements.txt
    ```

    **If you plan to use `Phi-4-multimodal-instruct`:**
    *Due to a `transformers` version incompatibility, please use this specific requirements file.*
    ```bash
    pip install -r requirements_phi4.txt
    ```
    *Note: `fitz` requires PyMuPDF.*

3.  **API Keys (for GPT-4o):**
    If you plan to use the GPT-4o model, make sure to set your OpenAI API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```

4.  **Download Datasets and Models:**
    The scripts assume that datasets are placed in a `data/` directory. You can download the public benchmarks from the following links:
    -   [**CORD-v2**](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
    -   [**DocVQA**](https://www.docvqa.org/datasets/docvqa)
    -   [**SROIE**](https://rrc.cvc.uab.es/?ch=13)
    -   [**Kleister-Charity**](https://huggingface.co/datasets/applicaai/kleister-charity)

    For the models, please note that the scripts are configured with `local_files_only=True` by default when loading models from Hugging Face. This means you must download the model weights and files beforehand. You can do this by running a script that downloads and caches them once.

## ⚡ Running Inference

All inference scripts share a similar command-line interface. You need to specify the model you want to run and, optionally, the level of quantization.

**General Command Structure:**
```bash
python experiments/<script_name>.py --model_name <model_name> [--quantize <quantization>]
```

**Example: Running the CORD experiment with Qwen-VL**
```bash
python experiments/CORD.py --model_name "Qwen2.5-VL-3B-Instruct"
```

**Example: Running the DocVQA experiment with 4-bit quantization**
```bash
python experiments/DocVQA.py --model_name "InternVL3-2B-Instruct" --quantize "4-bit"
```

### Command-Line Arguments for Inference
-   `--model_name`: (Required) The name of the model to run. Choices: `"Qwen2.5-VL-3B-Instruct"`, `"InternVL3-2B-Instruct"`, `"Phi-4-multimodal-instruct"`, `"GPT-4o"`.
-   `--quantize`: (Optional) The quantization level. Choices: `"8-bit"`, `"4-bit"`. Omit for no quantization.
-   `--FT_root`: (Optional) Path to a trained LoRA module to be loaded for inference.
-   `--size`: (Optional, for `DocVQA` and `KLC`) The number of samples from the dataset to run.

### Output Location
The results for each run, including detailed JSON outputs and final scores, are saved in the `results/` directory. The output path is structured as follows: `results/<Dataset_Name>/<Quantization>/<Model_Name>/`.

## 🔥 Fine-Tuning Models

This project includes scripts for fine-tuning the `Qwen2.5-VL-3B-Instruct` model using LoRA on various datasets.

**General Command Structure:**
```bash
python experiments/<finetune_script_name>.py [arguments]
```

**Example: Fine-tuning on the CORD dataset**
```bash
python experiments/FT_qwen.py \
    --quantize "4-bit" \
    --epochs 3 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --r 8 \
    --alpha 16 \
    --output_dir "qwen_lora_cord"
```

**Example: Fine-tuning on a combination of custom datasets (D1 and D2)**
```bash
python experiments/FT_qwen_D12.py \
    --quantize "4-bit" \
    --epochs 3 \
    --train_size 1000 \
    --eval_size 100 \
    --output_dir "qwen_lora_d12"
```

### Command-Line Arguments for Fine-Tuning
-   `--quantize`: (Optional) Quantization level (`"8-bit"`, `"4-bit"`).
-   `--epochs`: Number of training epochs.
-   `--train_batch_size`: Batch size for training.
-   `--eval_batch_size`: Batch size for evaluation.
-   `--r`: The rank for LoRA.
-   `--alpha`: The alpha parameter for LoRA.
-   `--output_dir`: Directory to save the trained LoRA adapter.
-   `--train_size`, `--eval_size`: (Optional) Number of samples to use for training/evaluation.

## 📂 Available Experiments

-   `CORD.py`: Inference on the CORD dataset.
-   `DocVQA.py`: Inference on the DocVQA dataset.
-   `SROIE.py`: Inference on the SROIE dataset.
-   `KLC.py`: Inference on the Kleister-Charity dataset.
-   `dataset1.py`, `dataset2.py`, `dataset3.py`: Inference on custom datasets.
-   `FT_qwen.py`: Fine-tuning on CORD.
-   `FT_qwen_D1.py`, `FT_qwen_D2.py`, `FT_qwen_D3.py`: Fine-tuning on individual custom datasets.
-   `FT_qwen_D12.py`, `FT_qwen_D123.py`: Fine-tuning on combined custom datasets.

## 🛠️ Utility Scripts

The `utils/` directory contains helper functions used across the experiments:
-   `CORD_util.py`: Evaluation logic (Tree Edit Distance) for the CORD dataset.
-   `FT_util.py`: Helper functions for fine-tuning.
-   `Internvl_util.py`: Image preprocessing specific to InternVL.
-   `KLC_util.py`: Confidence score calculation for the KLC experiment.
-   `LLM_util.py`: Utilities for using LLMs (like GPT-4o) as judges.
-   `util.py`: General utility functions, including the ANLS score.