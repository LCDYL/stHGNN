# stHGNN

**stHGNN: Deciphering Spatial Transcriptomics Data via Dual Hypergraph Learning Enhancement**

## Overview

stHGNN is a dual-view hypergraph neural network for spatial domain identification in spatial transcriptomics (ST). It models higher-order cellular dependencies from gene-expression and spatial views, integrates complementary information through cross-attention and global self-correlation reorganization, and jointly optimizes hypergraph smoothing, ZINB reconstruction, and multi-view self-supervised clustering.

The method is evaluated on the human dorsolateral prefrontal cortex (DLPFC), human breast cancer (HBC), and mouse anterior brain (MAB) datasets. The current repository provides a DLPFC preprocessing and training example.

## Repository Structure

```text
stHGNN/
├── DLPFC_test.py                 # Training and evaluation entry
├── P_G_model.py                  # stHGNN architecture
├── input_my_data.py              # Data loading and hypergraph construction
├── loss.py                       # Loss functions
├── config.py                     # Configuration parser
├── DLPFC.ini                     # Experimental parameters
├── process_data/
│   └── DLPFC_generate_data.py    # DLPFC preprocessing
├── requirements.txt              # Python dependencies
└── LICENSE
```

## Installation

Python 3.8 is recommended. Install the core packages used in this project as follows:

```bash
conda create -n sthgnn python=3.8 -y
conda activate sthgnn

pip install torch==2.4.1 torchvision==0.19.1
pip install torch-geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18

pip install scanpy==1.9.8 anndata==0.9.2
pip install numpy==1.22.4 pandas==2.0.3 scipy==1.10.1
pip install scikit-learn==1.3.2 h5py==3.1.0
pip install matplotlib==3.7.5 seaborn==0.13.2 umap-learn==0.5.7
```

For CUDA environments, install PyTorch and the PyTorch Geometric extensions using versions compatible with the local CUDA toolkit.
## Data Preparation

Download the DLPFC Visium data and the corresponding `metadata.tsv`, and then configure the local data path in `process_data/DLPFC_generate_data.py`.

```bash
cd process_data
python DLPFC_generate_data.py
cd ..
```

The processed data are saved to:

```text
generate_data/DLPFC/<slice_id>/<slice_id>.h5ad
```

For custom data, the AnnData object should contain:

```text
X
obsm["spatial"]
obs["ground"]
obs["ground_truth"]
```

The benchmark script uses the annotation fields to determine the cluster number and calculate ARI/NMI.

## Usage

Set the target slice and processed-data path in `DLPFC_test.py`, for example:

```python
datasets = ["151671"]
H5AD_PATH = f"generate_data/DLPFC/{datasets[i]}/{datasets[i]}.h5ad"
```

Run the experiment:

```bash
python DLPFC_test.py
```

Hyperparameters are specified in `DLPFC.ini`. The current scripts use CPU by default; modify the device definitions in `DLPFC_test.py` and `P_G_model.py` to enable CUDA.

> Note: several configuration identifiers use `MBA` to denote the MAB dataset.

## Output

Results are saved under:

```text
result/DLPFC/<slice_id>/
```

The outputs include spatial-domain labels, latent embeddings, reconstructed expression, an AnnData file, and spatial/UMAP visualizations.

## Citation

If you use this work, please cite:

```bibtex
@article{tang2026sthgnn,
  title={stHGNN: Deciphering spatial transcriptomics data via dual hypergraph learning enhancement},
  author={Tang, Jinghong and Chen, Lezhi and Yi, Siyu and Liu, Wei and Li, Mingyang and Wang, Yifan and Qiao, Ziyue and Guo, Bin and Liu, Xianggen and Lv, Jiancheng and others},
  journal={Pattern Recognition},
  pages={114585},
  year={2026},
  publisher={Elsevier}
}
```
