# Lianhuanhua Visual Analysis with Contrastive Learning

This repository is a Proof of Concept (PoC) for analyzing Lianhuanhua using a visual presentation model built on contrastive learning principles.
The project is structured to facilitate a step-by-step workflow from data preparation to result visualization.

## Getting Started

To get started with this project, clone the repository and install the required dependencies.

### Prerequisites

- Python 3.10 or higher
- Pip package manager

---
### Installation

1. Clone the repository:
```bash
git https://github.com/MeeBoo001/contrastive_lianhuanhua.git
cd contrastive_lianhuanhua
```

2. Install dependencies:
```bash
python -m venv lhh_env
source lhh_env/bin/activate
pip install -r requirements.txt
```

3. Download data:
The data of covers of Lianhuanhua are available at [here](https://drive.google.com/file/d/15uXnywDU9JgLUaLMAsLSYxnbacu65yM5).

You may need to download the data and unzip all images in `./data/`

---
### Usage
Execute the scripts in the numerical order provided to perform the complete analysis:

1. Generate annotations for the dataset:
```bash
python 1-annotation_generate.py
```

2. Fine-tune the contrastive model:
```bash
python 2-fine-tuning.py
```

Note: If you do not have enough GPU resources to fine-tune the model,
you can skip this step.

3. Extract features from the model:
```bash
python 3-feature_extract.py
```
Note: If no fine-tuned model is found, the script will use a ResNet18 model
pretrained on [ImageNet](https://image-net.org/) to extract visual features.

4. Cluster extracted features for analysis:
```bash
python 4-feature_cluster.py
```

5. Visualize the results:
```bash
python 5-results_visualise.py
```

Each script can be run independently, provided that the previous steps have been completed.

---
### Sample Images from the Clusters
Samples in each row are clustered into one cluster with K-means algorithm, with visual representation extracted from the trained SimCLR model. 
![Lianhuanhua Sample](cluster_samples0.jpg)
