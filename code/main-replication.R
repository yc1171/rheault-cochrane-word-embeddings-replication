# Script 1: Main Replication - Rheault & Cochrane (2020)
# This script extracts embeddings from pre-trained models, runs PCA,
# creates visualizations, and validates results against external benchmarks

# ---- Environment Setup ----
library(reticulate)
# Configure Python environment
use_python("~/.pyenv/versions/partyembed38/bin/python", required = TRUE)
# Show Python configuration
py_config()

# ---- Load Libraries ----
library(tidyverse)
library(ggrepel)
library(ggplot2)
library(readr)
library(stringr)
library(dplyr)

# ---- US House Analysis ----
# Extract vectors from the pre-trained US House model
extract_us_house_vectors <- function() {
  py_run_string('
  from gensim.models import Doc2Vec
  import pandas as pd
  import numpy as np
  import os
  
  # Load the pre-trained model
  model = Doc2Vec.load("models/house200")
  
  # Extract party tags (D_ for Democrats, R_ for Republicans)
  all_tags = list(model.docvecs.doctags.keys())
  party_tags = [tag for tag in all_tags if str(tag).startswith(("D_", "R_"))]
  print(f"Found {len(party_tags)} party tags")
  
  # Extract embeddings for party tags
  party_vectors = [model.docvecs[tag] for tag in party_tags]
  
  # Create year mapping from congress numbers
  tag_years = []
  for tag in party_tags:
      tag_str = str(tag)
      parts = tag_str.split("_")
      if len(parts) > 1 and parts[1].isdigit():
          congress = int(parts[1])
          year = 1789 + (congress - 1) * 2  # Convert congress number to approximate year
          party = "Democrats" if parts[0] == "D" else "Republicans"
          tag_years.append({
              "tag": tag_str, 
              "party": party, 
              "congress": congress, 
              "year": year
          })
  
  # Save vectors and metadata to CSV files
  os.makedirs("data", exist_ok=True)
  party_df = pd.DataFrame(party_vectors, index=party_tags)
  party_df.to_csv("data/us_house_vectors.csv")
  pd.DataFrame(tag_years).to_csv("data/us_house_tag_years.csv", index=False)
  print(f"Saved {len(party_vectors)} vectors and {len(tag_years)} tag-year mappings")
  ')
}

# Run PCA and visualize US House party placements
analyze_us_house <- function() {
  # Load the extracted embeddings and metadata
  party_vectors <- read_csv("data/us_house_vectors.csv", show_col_types = FALSE)
  tag_years <- read_csv("data/us_house_tag_years.csv", show_col_types = FALSE)
  
  # Prepare data for PCA
  mat <- as.matrix(party_vectors[,-1])
  rownames(mat) <- party_vectors[[1]]
  
  # Run PCA to reduce to two dimensions
  pca <- prcomp(mat, scale. = TRUE)
  pca_df <- data.frame(party_tag = rownames(mat), PC1 = pca$x[,1], PC2 = pca$x[,2])
  
  # Join with year and party information
  pca_df <- left_join(pca_df, tag_years, by = c("party_tag" = "tag"))
  
  # Reorient axes to match conventional interpretation
  # Democrats should be on the left (negative PC1)
  if(mean(pca_df$PC1[pca_df$party == "Democrats"]) > mean(pca_df$PC1[pca_df$party == "Republicans"])) {
    pca_df$PC1 <- -pca_df$PC1
  }
  
  # For PC2, recent Democrats should be above Republicans (South-North orientation)
  recent_dem <- pca_df %>% filter(party == "Democrats", year > 2000)
  recent_rep <- pca_df %>% filter(party == "Republicans", year > 2000)
  if(mean(recent_dem$PC2) < mean(recent_rep$PC2)) {
    pca_df$PC2 <- -pca_df$PC2
  }
  
  # Select key years to label in the plot
  key_years <- c(1873, 1903, 1927, 1933, 1945, 1957, 1969, 1975, 1981, 1987, 1993, 1999, 2005, 2011)
  label_points <- pca_df %>%
    filter(year %in% key_years) %>%
    mutate(label = paste0(substr(party, 1, 3), " ", year))
  
  # Set visual attributes
  party_colors <- c("Democrats" = "#3366CC", "Republicans" = "#CC3333")
  party_shapes <- c("Democrats" = 19, "Republicans" = 15)
  
  # Panel (a): Two-dimensional projection
  p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = party, shape = party)) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_text_repel(
      data = label_points,
      aes(label = label),
      size = 2.5,
      box.padding = 0.5,
      point.padding = 0.3,
      segment.color = "gray50",
      max.overlaps = 30
    ) +
    scale_color_manual(values = party_colors) +
    scale_shape_manual(values = party_shapes) +
    labs(
      title = "(a) Two-Dimensional Projection",
      x = "Component 1",
      y = "Component 2"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
    )
  
  # Create time series data
  time_series <- pca_df %>%
    arrange(year) %>%
    group_by(party, year) %>%
    summarize(
      PC1 = mean(PC1),
      PC2 = mean(PC2),
      .groups = 'drop'
    )
  
  # Panel (b): First dimension time series
  p2 <- ggplot(time_series, aes(x = year, y = PC1, color = party, linetype = party)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = party_colors) +
    scale_linetype_manual(values = c("Democrats" = "dashed", "Republicans" = "solid")) +
    labs(
      title = "(b) First Dimension",
      x = "Year",
      y = "Ideological Placement\n(First Principal Component)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "upper left",
      legend.title = element_blank(),
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
    )
  
  # Panel (c): Second dimension time series
  p3 <- ggplot(time_series, aes(x = year, y = PC2, color = party, linetype = party)) +
    geom_line(linewidth = 1) +
    scale_color_manual(values = party_colors) +
    scale_linetype_manual(values = c("Democrats" = "dashed", "Republicans" = "solid")) +
    labs(
      title = "(c) Second Dimension",
      x = "Year",
      y = "South-North Axis\n(Second Principal Component)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "lower right",
      legend.title = element_blank(),
      plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
    )
  
  # Return the plots
  return(list(p1=p1, p2=p2, p3=p3))
}

# ---- UK Parliament Analysis ----
# Extract vectors from the pre-trained UK model
extract_uk_vectors <- function() {
  py_run_string('
  from gensim.models import Doc2Vec
  import pandas as pd
  
  # Load the model
  model = Doc2Vec.load("models/uk200")
  
  # Get all document tags (e.g., "labour_1997", etc.)
  party_keys = list(model.docvecs.doctags.keys())
  
  # Extract vectors
  party_vectors = [model.docvecs[k] for k in party_keys]
  
  # Save to CSV for use in R
  party_df = pd.DataFrame(party_vectors, index=party_keys)
  party_df.to_csv("data/uk_party_vectors.csv")
  ')
}

# Run PCA and visualize UK party placements
analyze_uk_parliament <- function() {
  # Load the party vectors
  party_vectors <- read_csv("data/uk_party_vectors.csv")
  
  mat <- as.matrix(party_vectors[,-1])
  rownames(mat) <- party_vectors[[1]]
  
  # Run PCA
  pca <- prcomp(mat, scale. = TRUE)
  pca_df <- data.frame(party_tag = rownames(mat), PC1 = pca$x[,1], PC2 = pca$x[,2])
  
  # Extract party and year from tags
  pca_df <- pca_df %>%
    mutate(
      party = case_when(
        str_detect(tolower(party_tag), "lab") ~ "Labour",
        str_detect(tolower(party_tag), "con") ~ "Conservative",
        str_detect(tolower(party_tag), "lib|ldem") ~ "Liberal-Democrat",
        TRUE ~ "Other"
      ),
      year = as.numeric(str_extract(party_tag, "\\d{4}"))
    ) %>%
    filter(party != "Other")  # Focus on the three main parties
  
  # For labeling, select only a few significant points
  label_points <- pca_df %>%
    group_by(party) %>%
    filter(year %in% c(1979, 1997, 2001, 2005, 2010)) %>%
    mutate(label = paste0(party, " ", year))
  
  # Create color palette
  party_colors <- c("Labour" = "#E41A1C", "Conservative" = "#377EB8", "Liberal-Democrat" = "#FFD700")
  
  # Create the plot
  # need to multiply by -1 to match paper orientation
  pca_df$PC1 <- -pca_df$PC1 
  pca_df$PC2 <- -pca_df$PC2 
  
  # Plot
  p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = party, shape = party)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_text_repel(
      data = label_points,
      aes(label = label),
      size = 3.5,
      max.overlaps = 15,
      box.padding = 0.5,
      point.padding = 0.3,
      force = 5
    ) +
    scale_color_manual(values = party_colors) +
    scale_shape_manual(values = c(16, 17, 15)) +  # Different shapes for each party
    labs(
      title = "Party Placement in Britain (1935-2014)",
      x = "Component 1",
      y = "Component 2"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "gray", fill = NA, size = 0.5)
    )
  
  return(p)
}

# ---- Validation Analysis ----
# Function to calculate pairwise accuracy
calc_pairwise_accuracy <- function(actual, predicted) {
  # Remove NA values
  valid <- !is.na(actual) & !is.na(predicted)
  actual <- actual[valid]
  predicted <- predicted[valid]
  
  # Need at least 2 points to form a pair
  if(length(actual) < 2) return(NA)
  
  n <- length(actual)
  pairs <- 0
  correct_pairs <- 0
  
  for(i in 1:(n-1)) {
    for(j in (i+1):n) {
      pairs <- pairs + 1
      if((actual[i] < actual[j] && predicted[i] < predicted[j]) ||
         (actual[i] > actual[j] && predicted[i] > predicted[j])) {
        correct_pairs <- correct_pairs + 1
      }
    }
  }
  
  return(100 * correct_pairs / pairs)
}

# Validate against external benchmarks
validate_us_house <- function() {
  # Extract and load PCA results
  party_vectors <- read_csv("data/us_house_vectors.csv", show_col_types = FALSE)
  tag_years <- read_csv("data/us_house_tag_years.csv", show_col_types = FALSE)
  
  mat <- as.matrix(party_vectors[,-1])
  rownames(mat) <- party_vectors[[1]]
  
  pca <- prcomp(mat, scale. = TRUE)
  pca_df <- data.frame(party_tag = rownames(mat), PC1 = pca$x[,1], PC2 = pca$x[,2])
  pca_df <- left_join(pca_df, tag_years, by = c("party_tag" = "tag"))
  
  # Reorient PC1 to match conventional ideology (Democrats negative, Republicans positive)
  if(mean(pca_df$PC1[pca_df$party == "Democrats"]) > mean(pca_df$PC1[pca_df$party == "Republicans"])) {
    pca_df$PC1 <- -pca_df$PC1
  }
  
  # Load DW-NOMINATE scores (roll-call votes)
  dw_scores <- read_csv("data/dw_nominate_house.csv", show_col_types = FALSE)
  
  # Match congress numbers
  validation_df <- left_join(
    pca_df %>% 
      select(party, congress, PC1) %>% 
      group_by(party, congress) %>% 
      summarize(PC1 = mean(PC1), .groups = "drop"),
    dw_scores,
    by = c("party", "congress")
  )
  
  # Calculate correlation and pairwise accuracy
  voteview_correlation <- cor(validation_df$PC1, validation_df$dwnom1, use = "pairwise.complete.obs")
  voteview_accuracy <- calc_pairwise_accuracy(validation_df$dwnom1, validation_df$PC1)
  
  # Load expert survey data
  expert_surveys <- read_csv("data/expert_surveys_us.csv", show_col_types = FALSE)
  
  # Match with our PCA results
  expert_validation <- left_join(
    pca_df %>% 
      filter(year %in% expert_surveys$year) %>%
      select(party, year, PC1) %>%
      group_by(party, year) %>%
      summarize(PC1 = mean(PC1), .groups = "drop"),
    expert_surveys,
    by = c("party", "year")
  )
  
  # Calculate correlation and accuracy
  expert_correlation <- cor(expert_validation$PC1, expert_validation$expert_score, use = "pairwise.complete.obs")
  expert_accuracy <- calc_pairwise_accuracy(expert_validation$expert_score, expert_validation$PC1)
  
  # Load CMP manifesto data
  cmp_data <- read_csv("data/cmp_us.csv", show_col_types = FALSE)
  
  # Match with our PCA results (using election year that starts a congress)
  cmp_validation <- left_join(
    pca_df %>%
      filter(year %in% cmp_data$year) %>%
      select(party, year, PC1) %>%
      group_by(party, year) %>%
      summarize(PC1 = mean(PC1), .groups = "drop"),
    cmp_data,
    by = c("party", "year")
  )
  
  # Calculate correlations and accuracies for each CMP measure
  rile_corr <- cor(cmp_validation$PC1, cmp_validation$rile, use = "pairwise.complete.obs")
  rile_accuracy <- calc_pairwise_accuracy(cmp_validation$rile, cmp_validation$PC1)
  
  vanilla_corr <- cor(cmp_validation$PC1, cmp_validation$vanilla, use = "pairwise.complete.obs")
  vanilla_accuracy <- calc_pairwise_accuracy(cmp_validation$vanilla, cmp_validation$PC1)
  
  legacy_corr <- cor(cmp_validation$PC1, cmp_validation$legacy, use = "pairwise.complete.obs")
  legacy_accuracy <- calc_pairwise_accuracy(cmp_validation$legacy, cmp_validation$PC1)
  
  # Create results table
  results <- data.frame(
    Gold_Standard = c("Voteview", "Expert surveys", "Rile", "Vanilla", "Legacy"),
    Correlation = c(voteview_correlation, expert_correlation, rile_corr, vanilla_corr, legacy_corr),
    Pairwise_Accuracy = c(voteview_accuracy, expert_accuracy, rile_accuracy, vanilla_accuracy, legacy_accuracy)
  )
  
  return(results)
}

# Validate UK results
validate_uk_parliament <- function() {
  # Load the UK party vectors
  party_vectors <- read_csv("data/uk_party_vectors.csv")
  mat <- as.matrix(party_vectors[,-1])
  rownames(mat) <- party_vectors[[1]]
  
  # Run PCA
  pca <- prcomp(mat, scale. = TRUE)
  pca_df <- data.frame(party_tag = rownames(mat), PC1 = pca$x[,1], PC2 = pca$x[,2])
  
  # Extract party and year info
  pca_df <- pca_df %>%
    mutate(
      party = case_when(
        str_detect(tolower(party_tag), "lab") ~ "Labour",
        str_detect(tolower(party_tag), "con") ~ "Conservative",
        str_detect(tolower(party_tag), "lib|ldem") ~ "Liberal-Democrat",
        TRUE ~ "Other"
      ),
      year = as.numeric(str_extract(party_tag, "\\d{4}"))
    ) %>%
    filter(party != "Other")
  
  # Orient PC1 so Labour is negative, Conservative is positive
  if(mean(pca_df$PC1[pca_df$party == "Labour"]) > mean(pca_df$PC1[pca_df$party == "Conservative"])) {
    pca_df$PC1 <- -pca_df$PC1
  }
  
  # Load gold standard data
  gold_uk <- read_csv("data/goldstandard_uk.csv")
  
  # Create matching labels for joining
  pca_df$gold_label <- paste(pca_df$party, pca_df$year)
  
  # Merge with gold standard
  validation_df <- left_join(
    gold_uk, 
    pca_df %>% select(gold_label, PC1), 
    by = c("label" = "gold_label")
  )
  
  # Calculate correlations and accuracies
  results <- data.frame(
    Gold_Standard = c("Experts Survey", "Rile", "Vanilla", "Legacy"),
    Correlation = NA,
    Pairwise_Accuracy = NA,
    stringsAsFactors = FALSE
  )
  
  # Calculate correlations and accuracies
  gold_cols <- c("experts_stand", "rile", "vanilla", "legacy")
  for(i in 1:length(gold_cols)) {
    col <- gold_cols[i]
    
    # Only calculate if we have enough data points
    valid_data <- !is.na(validation_df[[col]]) & !is.na(validation_df$PC1)
    if(sum(valid_data) >= 3) {  # Need at least 3 points for meaningful correlation
      results$Correlation[i] <- cor(validation_df[[col]][valid_data], 
                                    validation_df$PC1[valid_data])
      
      if(sum(valid_data) >= 2) {  # Need at least 2 points for pairwise accuracy
        results$Pairwise_Accuracy[i] <- calc_pairwise_accuracy(
          validation_df[[col]], validation_df$PC1)
      }
    }
  }
  
  return(results)
}


# ---- Run the complete analysis ----
main <- function() {
  # Extract embeddings from models
  extract_us_house_vectors()
  extract_uk_vectors()
  
  # Run PCA and create plots
  us_plots <- analyze_us_house()
  uk_plot <- analyze_uk_parliament()
  
  # Validate results
  us_validation <- validate_us_house()
  uk_validation <- validate_uk_parliament()
  
  # Save results
  dir.create("output", showWarnings = FALSE)
  pdf("output/us_house_plots.pdf", width = 12, height = 10)
  gridExtra::grid.arrange(us_plots$p1, us_plots$p2, us_plots$p3, ncol = 2)
  dev.off()
  
  ggsave("output/uk_parliament_plot.pdf", uk_plot, width = 8, height = 7)
  
  write.csv(us_validation, "output/us_validation_results.csv", row.names = FALSE)
  write.csv(uk_validation, "output/uk_validation_results.csv", row.names = FALSE)
  
  pdf("output/wordfish_comparison.pdf", width = 10, height = 8)
  gridExtra::grid.arrange(wordfish_comparison$full_corpus, wordfish_comparison$recent_years, ncol = 1)
  dev.off()
  
  # Print summary
  cat("==== Replication Summary ====\n")
  cat("US House Validation Results:\n")
  print(us_validation)
  cat("\nUK Parliament Validation Results:\n")
  print(uk_validation)
}

# Run the main analysis
main()