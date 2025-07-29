# Enterprise Risk Prediction with Heterogeneous Graph Attention Networks (HAG-Net)

## 📖 Overview

This repository contains the official implementation for the research project **"Enterprise Risk Prediction with Heterogeneous Graph Attention Networks"**. This project aims to predict the credit risk of enterprises by modeling the complex web of relationships they are embedded in, including guarantee chains, investment ties, supply chains, and market correlations.

Our core contribution is **HAG-Net**, a hierarchical attention graph network designed specifically for heterogeneous financial graphs. It leverages a multi-level attention mechanism to capture both the importance of individual neighbors (node-level attention) and the significance of different relationship types (relation-level attention), providing a more accurate and interpretable risk assessment.

This work builds upon the foundational ideas of models like HAN (Heterogeneous Graph Attention Network) and the risk assessment framework proposed by Bi et al. (2024), extending them by incorporating more diverse financial relationships and a more robust heterogeneous structure that includes both company and person nodes.

## ✨ Key Features

- **Heterogeneous Graph Construction**: Scripts and methodologies to build a complex financial graph with multiple node types (`Company`, `Person`) and edge types (`Guarantees`, `Invests`, `Supply`, `Market_Correlation`).
- **Hierarchical Attention Model (HAG-Net)**: A PyTorch implementation of our proposed HAG-Net model, built using the PyTorch Geometric (PyG) library.
- **Feature Engineering**: Includes modules for processing time-series financial data into meaningful features for the static graph model.
- **Interpretability Analysis**: Tools to visualize attention weights at both the node and relation levels, helping to uncover risk transmission mechanisms.
- **Reproducibility**: Complete pipeline for data processing, model training, evaluation, and baseline comparisons.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/HAG-Net-Enterprise-Risk.git
    cd HAG-Net-Enterprise-Risk
    ```

2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n hag_net python=3.9
    conda activate hag_net
    ```

3.  **Install dependencies:**
    This project relies heavily on PyTorch and PyTorch Geometric. Please follow the official PyG installation instructions that match your CUDA version.

    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision torchaudio
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    pip install pyg_lib torch_geometric
    
    # Install other packages
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` should include pandas, numpy, scikit-learn, xgboost, etc.*

## 📂 Repository Structure
