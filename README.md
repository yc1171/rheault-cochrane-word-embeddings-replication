# README: Main Replication Script for Rheault & Cochrane (2020) Replication

This repository contains the main replication code for the paper **"Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora"** by Rheault and Cochrane (2020). The script extracts word embeddings from pre-trained models, performs PCA, visualizes party placements, and validates ideological scores against multiple external benchmarks.

## Overview

The paper explores whether neural network-based word embeddings, augmented with political metadata, can effectively capture latent ideological positions in political text. Using parliamentary speeches from the **United States**, **Britain**, and **Canada**, it trains models with party-year labels to produce "party embeddings." These embeddings are analyzed using PCA and validated against external indicators like roll-call votes, expert surveys, and the Comparative Manifesto Project (CMP).

Our script replicates the **U.S. House of Representatives** and **U.K. Parliament** components of the original paper.

### Key Replications:

-   **U.S. House (1873--2016)**: PCA identifies a clear left-right ideological dimension (PC1) and a regional South-North realignment (PC2).
-   **U.K. Parliament (1935--2014)**: PCA distinguishes Labour (left), Conservatives (right), and Liberal Democrats (center), reproducing shifts under Blair and Thatcher. The second component loosely corresponds to government-opposition dynamics.

### Validation:

The script replicates benchmark comparisons: - **Voteview DW-NOMINATE scores** - **Expert survey ideology scores** - **CMP Manifesto indicators** (RILE, Vanilla, Legacy)

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

## Data Needed

The following files must be present in the `data/` directory:

-   `dw_nominate_house.csv`
-   `expert_surveys_us.csv`
-   `cmp_us.csv`
-   `goldstandard_uk.csv`

The models (`house200`, `uk200`) should be placed in the `models/` folder.

## Citation

If you use this code, please cite:

> Rheault, L., & Cochrane, C. (2020). *Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora*. Political Analysis, 28(1), 112--133.
>
> 
---

## Extension: Analyzing Congressional Tweets with Word Embeddings

In addition to replicating the original paper’s results using parliamentary speech, this project includes an **extension using U.S. Congressional Twitter data**.

### Motivation

This extension tests whether word embeddings derived from **informal political text** (tweets) can reveal similar ideological patterns as formal legislative speeches.

### Method

- **Corpus**: Tweets from U.S. Congress members.
- **Model**: A custom-trained Doc2Vec model using tweet text.
- **User Embedding**: Each legislator's tweets are averaged to create a single "user vector."
- **PCA**: Principal Component Analysis is applied to user vectors to visualize ideological and stylistic variation.

### Key Findings

- **PC1 (Ideology)**: The first principal component captures the **left-right political spectrum** among legislators.
- **PC2 (Style)**: The second component captures stylistic variation (e.g., personal vs. policy tone).
- **Word Projections**: Words most strongly associated with ideological extremes are identified using PCA loadings.

For example:
- Right-end words: `"demtaxhikes"`, `"huckabee"`, `"deductio"`
- Left-end words: `"covid"`, `"help"`, `"americans"`

### Limitations & Considerations

- Results are more **noisy** than the original paper due to limited preprocessing.
- **GloVe-style co-occurrence vectors** are less tailored than party-specific embeddings.
- Cleaning and metadata-aware modeling could improve future performance.

### Value

This shows that **unsupervised embedding models can extract ideological dimensions** even from short, informal texts like tweets, suggesting a broader applicability of these methods beyond formal parliamentary debate.

