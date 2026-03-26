# A Model-Agnostic Re-ranking Framework for Creativity-Oriented Recommendations

## Overview
Modern Recommender Systems (RecSys) have achieved excellent accuracy in predicting user preferences. However, exclusively optimizing for accuracy—the *accuracy-only* paradigm—generates undesirable side effects, most notably the Filter Bubble. This phenomenon confines users within a "comfort zone" of predictable and redundant suggestions. Over time, this lack of novelty compromises long-term user satisfaction and retention.

This project aims to overcome these limitations by integrating principles of Computational Creativity (referencing Margaret Boden's theoretical framework) into the recommendation pipeline. The goal is to engineer *Serendipity*, balancing the relevance of suggestions with Novelty and Unexpectedness to offer recommendations that are not only useful but also original and surprising.

From a methodological standpoint, the codebase focuses on a specific re-ranking strategy applied downstream of standard retrieval models: the **Creativity Score Re-ranking**. This composite metric linearly aggregates relevance, novelty (based on inverse popularity), and unexpectedness (based on semantic distance), allowing for direct optimization of the creative potential of the recommendations. 

Experimental validation is built on top of the RecBole framework. The repository contains scripts to test collaborative filtering models and knowledge-aware architectures. Results show that while there is an inevitable trade-off between accuracy and creativity, the proposed strategy significantly boosts the novelty and unexpectedness of recommendations, mitigating the filter bubble without drastically degrading perceived system quality.

## Installation & Setup

To run the codebase, it is highly recommended to use an isolated Python virtual environment. 

Open the terminal:

### Windows Setup

```powershell
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

### Linux / macOS Setup

```bash
# 1. Create the virtual environment
python3 -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

## Metrics and Evaluation

In addition to standard accuracy metrics provided by the RecBole backend, the framework implements several custom metrics tailored specifically to measure the *creative* potential of the recommendations. 

### Custom Creativity Metrics

**1. Item Novelty**
A score inversely proportional to the item's occurrences within the training set. It rewards the system for surfacing long-tail items.
$$ \text{Novelty}(i) = \frac{1}{\log(1 + avgpop(i))} $$
*(where $avgpop(i)$ is the popularity count of item $i$)*

**2. Item Unexpectedness**
Measures how semantically different a candidate item is relative to the user's historical profile ($H_u$). It uses the average cosine similarity between the candidate's embedding and the embeddings of the items the user previously interacted with. The similarity is normalized from $[-1, 1]$ to $[0, 1]$ before being inverted, so that highly dissimilar items approach an unexpectedness of $1$.
$$ \text{sim}_{norm}(i, H_u) = \frac{\overline{\text{cosine\_sim}}(i, H_u) + 1}{2} $$
$$ \text{Unexpectedness}(i, u) = 1 - \text{sim}_{norm}(i, H_u) $$

**3. Creativity Score (Re-ranking Objective)**
The composite aggregated metric used during the re-ranking phase. It combines min-max normalized relevance (from base predictions), novelty, and unexpectedness through customizable weights ($w_{rel}$, $w_{nov}$, $w_{unexp}$).
$$ \text{Creativity Score} = w_{rel} \cdot \widehat{\text{Rel}} + w_{nov} \cdot \widehat{\text{Nov}} + w_{unexp} \cdot \widehat{\text{Unexp}} $$

**4. Serendipity (Gérard-Evangelista)**
A binary serendipity representation counting serendipitous hits: recommendations that are successfully relevant (found in ground truth $T_u$) AND strictly unpopular (not belonging to the set of the most popular items $PM_K$). Averaged over users:
$$ \text{Serendipity}_{GE}(u) = \frac{\left|\{ i \in L_u \cap T_u \mid i \notin PM_K \}\right|}{K} $$

**5. Serendipity & Unexpectedness (Yan et al.)**
Evaluates surprise based purely on user-to-item spatial distance (dot product of normal embeddings). 
$$ \text{Unexp}_{Yan}(i, u) = 1 - \frac{\text{sim}(\vec{u}, \vec{i}) + 1}{2} $$
The serendipity score averages this unexpectedness strictly over overlapping hits:
$$ \text{Serendipity}_{Yan}(u) = \frac{1}{K} \sum_{i \in L_u \cap T_u} \text{Unexp}_{Yan}(i, u) $$

### Standard Base Metrics (RecBole)
- **NDCG (Normalized Discounted Cumulative Gain):** Measures ranking quality, heavily penalizing relevant items placed lower in the list.
- **Recall & Precision:** Standard retrieval accuracy metrics evaluating hits against the user's hidden items.
- **AveragePopularity:** Evaluates the aggregate popularity of the recommended lists. Lower values typically denote better novelty.
- **GiniIndex & ShannonEntropy:** General measurements of diversity and distribution equality across the catalog. Evaluates whether the model recommends a diverse set of items across the whole user base rather than exploiting a specific niche.

The project uses two primary datasets, both included in the repository.

| Dataset | Users | Items | Interactions | Notes |
|---------|-------|-------|--------------|-------|
| **MovieLens-1M** | 6,040 | 3,952 | 1,000,210 | Explicit ratings (1–5) |
| **Amazon-Books (Reduced)** | 1,760 | 78,142 | 887,367 | Filtered for users with ≥500 interactions (original dataset had ~51.3 million interactions) |

## Models and Hyperparameters

The repository contains scripts that configure and train a variety of baselines and state-of-the-art architectures using the RecBole backend. Below is a summary of the primary models used and their key hyperparameter configurations. 

### General Configuration (Global)
All models share a set of common training parameters unless specified otherwise:
- **Seed:** 2020 (for reproducibility)
- **Batch Size (Train/Eval):** 256
- **Stopping Step:** 3 (early stopping patience)
- **Evaluation Settings:** TO_RS, 80_10_10 split (80% train, 10% validation, 10% test)
- **Relevance Threshold:** Rating ≥ 3 (implicit conversion)

### Model-specific Hyperparameters

| Model | Hyperparameters | Description |
|-------|----------------|-------------|
| **Pop** | `epochs`: 1 | Simple popularity baseline. Requires no iteration. |
| **ItemKNN** | `epochs`: 10, `k`: 64 | K-Nearest Neighbors item-based baseline. `k` indicates the neighborhood size. |
| **BPR** | `epochs`: 10, `learning_rate`: 0.001, `embedding_size`: 64 | Bayesian Personalized Ranking. Standard collaborative filtering via matrix factorization. |
| **DMF** | `epochs`: 20, `learning_rate`: 0.001, `user_embedding_size`: 64, `item_embedding_size`: 64, `user_hidden_size_list`: [64, 64], `item_hidden_size_list`: [64, 64] | Deep Matrix Factorization. Employs multi-layer perceptrons (MLPs) over embeddings. |
| **MultiVAE** | `epochs`: 15, `learning_rate`: 0.001, `hidden_dimension`: [600, 200], `latent_dimension`: 200, `dropout_prob`: 0.5 | Variational Autoencoder for collaborative filtering. Uses a non-linear generative approach. |
| **LightGCN** | `epochs`: 20, `learning_rate`: 0.001, `embedding_size`: 64, `n_layers`: 2, `reg_weight`: 1e-4 | Lightweight Graph Convolutional Network. Employs a simplified GCN architecture focusing purely on neighborhood aggregation. |
| **ENMF** | `epochs`: 15, `learning_rate`: 0.001 | Efficient Neural Matrix Factorization. Focuses on efficient training without negative sampling. |
| **CFKG** | `epochs`: 20, `learning_rate`: 0.001, `embedding_size`: 64, `loss_type`: "BPR" | Collaborative Filtering over Knowledge Graphs. Fuses CF with KG embeddings optimization. |
| **CKE** | `epochs`: 20, `learning_rate`: 0.001, `embedding_size`: 64, `kg_embedding_size`: 64 | Collaborative Knowledge Base Embedding. Extracts structural knowledge using translation frameworks (like TransR). |
| **KGCN** | `epochs`: 20, `learning_rate`: 0.001, `embedding_size`: 64, `kg_embedding_size`: 64, `neighbor_sample_size`: 4, `n_iter`: 2, `aggregator_type`: "sum" | Knowledge Graph Convolutional Networks. Aggregates neighborhood info from the KG (`n_iter` specifies depth, `neighbor_sample_size` restricts sample width). |
| **KGNN-LS** | `epochs`: 20, `learning_rate`: 0.001, `embedding_size`: 64, `kg_embedding_size`: 64, `neighbor_sample_size`: 4, `n_iter`: 2, `ls_weight`: 0.5 | Knowledge-aware Graph Neural Networks with Label Smoothness. Regularizes via user-specific relation combinations. |

*Note: These hyperparameter sets are defined explicitly inside the `MODELS` dictionary inside `train_and_save_recs.py` and `train_and_save_recs_KG.py`. They can be freely modified prior to execution to experiment with performance variability.*

## File Descriptions and Usage Guide

This section details every script within the repository, its core purpose, how to execute it, and the configurable parameters (hyperparameters) commonly found at the top of the files.

### 1. Data Preprocessing Scripts
These utility scripts are primarily used to analyze and reduce massive datasets (like Amazon-Books) to create denser, more manageable sub-datasets for training.

- **`analyze_amazon_thresholds.py`**
  - **Purpose**: Performs exploratory data analysis to evaluate sparsity and interaction densities. It helps researchers decide where to cut the dataset by printing out statistics based on varying interaction thresholds.
  - **Execution**: `python analyze_amazon_thresholds.py`
  - **Modifiable Parameters**: Inside the script, you can specify lists of *user thresholds* to test and the *minimum item interactions* required for an item to be kept in the statistics.

- **`create_amazon_cut.py`**
  - **Purpose**: Actually performs the dataset pruning based on the chosen thresholds, stripping out inactive users and items, and outputting the final `.inter` file used for training by the RecBole engine.
  - **Execution**: `python create_amazon_cut.py`
  - **Modifiable Parameters**: You can adjust `USER_THRESHOLD` (e.g., set to 500 to strictly keep users with $\ge 500$ ratings) and `ITEM_THRESHOLD` directly inside the code to enforce the limits.

### 2. Training Scripts
These scripts initialize, train, and test validating baselines, automatically saving their states continuously.

- **`train_and_save_recs.py` & `train_and_save_recs_KG.py`**
  - **Purpose**: The former is used for standard collaborative models (e.g., BPR, LightGCN), and the latter for Knowledge Graph-aware models (e.g., KGCN, CKE). They compute embeddings, optimize the error functions, and save model checkpoints.
  - **Execution**: `python train_and_save_recs.py`
  - **Modifiable Parameters**:
    - `TOPKS`: A list specifying the cutoff lengths for training validation evaluation (e.g., `[5, 10]`).
    - `SAVE_TOPK`: Number of top recommended candidate items to generate and log.
    - `SEED`: Random initialization seed (default: `2020`).
    - `GLOBAL_CONFIG`: Controls base architectural behaviors (batch sizes, early stopping step, evaluation split `80_10_10`). *Note: It is highly recommended to leave `GLOBAL_CONFIG` and `SEED` entirely unchanged if the goal is to perfectly replicate the experimental results of the thesis.*

### 3. Evaluation & Re-ranking Scripts
These scripts load the pre-trained models, extract the recommended candidates, and mathematically re-rank them to break the accuracy-only paradigm using the Creativity Score metrics. 

- **`eval_creativity_score_reranking.py`** (and specific variants `eval_creativity_enmf.py`, `eval_creativity_lightgcn.py`)
  - **Purpose**: The core analytical phase of the project. Loads the base interaction network alongside pre-computed embeddings, calculates items' original relevance, baseline novelty, and spatial unexpectedness, and finally constructs the rearranged user lists.
  - **Execution**: `python eval_creativity_score_reranking.py`
  - **Modifiable Parameters**:
    - `TARGET_CKPTS`: Instructs the script on which specific model weights to load. Look inside the generated `saved/` directory. **Example**: If training generated `saved/LightGCN-Aug-10-2023.pth`, you should set `TARGET_CKPTS = {'LightGCN-Aug-10-2023.pth'}`. If you leave it as an empty set `set()`, the script will automatically iterate over *all* `.pth` files inside `saved/`.
    - `TOPKS`: The target cutoff lengths for the absolute final rearranged lists (e.g., `[10]`).
    - `CANDIDATE_KS`: The sizes of the initial candidate pool generated by the baseline to be passed into the re-ranker window (e.g., extracting 50 base-items `[50]` to find the 10 most unexpected).
    - **Configuration Weights**: The most important variables for the re-ranking formula. You can shift the framework's behavior by altering `WEIGHT_RELEVANCE`, `WEIGHT_NOVELTY`, and `WEIGHT_UNEXPECTEDNESS` (e.g., `0.33` each for a perfectly balanced approach, or tweaking them to selectively maximize Serendipity).
