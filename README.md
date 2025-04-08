# Word Embeddings for Ideological Placement: Replication and Extension

This repository contains the main replication code for the paper **"Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora"** by Rheault and Cochrane (2020, *Political Analysis*). The main script extracts word embeddings from pre-trained models, performs PCA, visualizes party placements, and validates ideological scores against multiple external benchmarks.

## Overview

The original paper explores whether neural network-based word embeddings, augmented with political metadata, can effectively capture latent ideological positions in political text. Using parliamentary speeches from the **United States**, **Britain**, and **Canada**, it trains models with party-year labels to produce "party embeddings." These embeddings are analyzed using PCA and validated against external indicators like roll-call votes, expert surveys, and the Comparative Manifesto Project (CMP).

Our project:

1. **Replicates** the core findings from the U.S. House and U.K. Parliament analyses
2. **Extends** the methodology to U.S. Congressional Twitter data using two approaches

## Repository Structure

- `code/`: Contains the main replication scripts and extension scripts, which include:
- `Main-Replication.R`: Replicates the paper's core findings
- `Extension-Using-GloVe.R`: Applies GloVe embeddings to congressional tweets
- `Extension-Using-Doc2Vec.R`: Implements a Doc2Vec approach similar to the original paper
- `data/`: Contains validation datasets and generated embeddings
- `models/`: Contains pre-trained embedding models
- `output/`: Visualizations and result tables

## Main Replication

Our replication focuses on two key components of the original paper:

- **U.S. House (1873-2016)**: We reproduce the left-right ideological dimension (PC1) and South-North regional realignment (PC2)
- **U.K. Parliament (1935-2014)**: We replicate the positioning of Labour, Conservatives, and Liberal Democrats, including historical shifts

The replication validates these results against external benchmarks including (a) **Voteview DW-NOMINATE scores**, (b)**Expert survey ideology scores**, and (c) **CMP manifesto indicators** (RILE, Vanilla, Legacy).

## Code Structure

The `Main-Replication.R` script includes the following core components:

### 1. Environment Setup

-   Uses `reticulate` to bridge R and Python.
-   Specifies Python path and verifies configuration.

### 2. Embedding Extraction

-   **`extract_us_house_vectors()`**: Loads the pre-trained `house200` Doc2Vec model and extracts party embeddings and metadata.
-   **`extract_uk_vectors()`**: Loads `uk200` model and extracts U.K. party-year embeddings.

### 3. PCA Analysis and Visualization

-   **`analyze_us_house()`**: Performs PCA on U.S. embeddings, orients axes, and creates:
    -   A 2D projection (party placements)
    -   PC1 and PC2 time-series plots
-   **`analyze_uk_parliament()`**: PCA and visualization for U.K. parties, highlighting ideological and temporal trends.

### 4. Validation

-   **`validate_us_house()`**: Correlates PCA-derived scores with Voteview, expert survey, and CMP scores.
-   **`validate_uk_parliament()`**: Validates U.K. projections using expert and manifesto data.

### 5. Output Generation

-   Saves:
    -   US plots as `output/us_house_plots.pdf`
    -   UK plot as `output/uk_parliament_plot.pdf`
    -   Validation results as CSV files
    -   (Optional) `wordfish_comparison.pdf`

### 6. `main()` Function

Runs the full pipeline: extraction → PCA → validation → export.


## Data Needed

The following files must be present in the `data/` directory:

-   `dw_nominate_house.csv`
-   `expert_surveys_us.csv`
-   `cmp_us.csv`
-   `goldstandard_uk.csv`

The models (`house200`, `uk200`) should be placed in the `models/` folder.


## Extension: Analyzing Congressional Tweets with Word Embeddings

In addition to replicating the original paper’s results using parliamentary speech, this project includes an **extension using U.S. Congressional Twitter data**.

### Motivation

This extension tests whether unsupervised word embeddings derived from **informal political text** (tweets) can reveal similar ideological patterns as formal legislative speeches, or if it could be used to derive other quantities of interests to explore dimensions of vairation in a text corpus.

### Method

- **Corpus**: Tweets from U.S. Congress members.
- **Model**: A custom-trained Doc2Vec model using tweet text.
- **User Embedding**: Each legislator's tweets are averaged to create a single "user vector."
- **PCA**: Principal Component Analysis is applied to user vectors to visualize ideological and stylistic variation.

### GloVe Embeddings
Trains a term co-occurrence based model and creates user embeddings by averaging word vectors.

### Doc2Vec PV-DM
Implements an approach similar to the original paper, directly incorporating user identity during embedding training.

### Key Findings

- In the GloVe model, PC1 reflects communication mode (routine vs. issue-specific), while PC2 captures focus (geographic vs. partisan).
- In the Doc2Vec model, PC1 and PC2 contrast along the line of Foreign language (Spanish words and references) vs. rural/agricultural language.
- Both extensions reveal that Twitter communication is structured along dimensions different from formal parliamentary speech, one that highlights **constituency demographics** and **communication style** rather than traditional partisan divisions.

## Requirements

### R Packages

``` r
install.packages(c("tidyverse", "readr", "dplyr", "ggplot2", "ggrepel", "stringr", "reticulate", "gridExtra"))
```

### Python Environment

Set up with Python 3.8+ and:

``` bash
pip install gensim pandas numpy
```

Ensure the script's `use_python()` line matches your Python environment path.

## Usage

1. Ensure all dependencies are installed
2. Place required data files in the `data/` directory
3. Run scripts in the following order:
   - `Main-Replication.R` for core paper replication
   - `Extension-Using-GloVe.R` and/or `Extension-Using-Doc2Vec.R` for extensions

## Citation

If you use this code, please cite:

> Rheault, L., & Cochrane, C. (2020). *Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora*. Political Analysis, 28(1), 112--133.
>
> 

---

```bibtex
@article{rheault2020word,
  title={Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora},
  author={Rheault, Ludovic and Cochrane, Christopher},
  journal={Political Analysis},
  volume={28},
  number={1},
  pages={112--133},
  year={2020}
}
