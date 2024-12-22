# Leveraging Graph Neural Networks for Multiple Instance Learning with MNIST Bags

In this notebook, we explore **Multiple Instance Learning (MIL)** using **Graph Neural Networks (GNNs)** to learn bag-level representations. Each bag of instances is treated as a graph, with individual instances as nodes and their relationships as edges. The goal is to leverage the structural information within the bags and use GNNs to learn meaningful embeddings for **bag-level classification**.

## Dataset
We use the MNIST Bags dataset, where each bag consists of multiple images (instances) of digits. The bag-level label indicates whether the bag contains the digit '9' or not.

## Contents
- Data Preprocessing
- Graph Construction
- Model Building with GNNs
- Training and Evaluation
- Results and Insights

## Requirements
- Python 3.x
- TensorFlow
- PyTorch Geometric
- Other dependencies listed in `requirements.txt`

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MIL-GNN-MNIST.git
    ```
2. Navigate to the project directory:
    ```bash
    cd MIL-GNN-MNIST
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the notebook:
    ```bash
    jupyter notebook Multiple_Instance_Learning_with_GNNs.ipynb
    ```

## Results
Our experiments demonstrate that using GNNs for Multiple Instance Learning on the MNIST Bags dataset effectively captures the structural information within bags, resulting in meaningful embeddings for bag-level classification.

## Conclusion
This project showcases the potential of Graph Neural Networks in addressing Multiple Instance Learning problems by exploiting the relational information within bags of instances.

