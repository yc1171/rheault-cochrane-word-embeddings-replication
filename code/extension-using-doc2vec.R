# =====================================================
# Replication Script 3: Extension using Doc2Vec PV-DM Approach
# =====================================================

#### This script:

#### 1. Sets up Python environment and loads necessary libraries for Doc2Vec modeling
#### 2. Preprocesses congressional tweets with comprehensive cleaning and stopword removal
#### 3. Trains a Doc2Vec model using the PV-DM (Distributed Memory) approach - similar to original paper
#### 4. Extracts user-level embeddings directly from the model
#### 5. Performs PCA on these user embeddings
#### 6. Identifies and visualizes extreme users and words on principal components
#### 7. Analyzes relationships between users and their similar words
#### 8. Compares results with the original paper's methodology

# =====================================================
#  STEP 0: Load packages & set up environment
# =====================================================

library(reticulate)
library(tidyverse)
library(ggplot2)
library(ggrepel)
library(stringr)
library(gridExtra)

# Configure Python environment - modify path as needed
use_python("~/.pyenv/versions/3.8.10/bin/python", required = TRUE)
py_config()

# =====================================================
#  STEP 1: Define Python preprocessing functions
# =====================================================

py_run_string('
import pandas as pd
import numpy as np
import re
import os
import emoji
import string
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Function to remove emojis
def remove_emoji(text):
    return emoji.replace_emoji(text, replace="")

# Function to clean text more thoroughly
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove @handles
    text = re.sub(r"@\w+", "", text)
    # Remove emojis
    text = remove_emoji(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Remove single letters
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
')

# =====================================================
#  STEP 2: Define Python functions for tweets processing
# =====================================================

py_run_string('
# Define function to load and preprocess tweets
def load_and_preprocess_tweets(csv_path):
    """
    Reads the tweet CSV, drops retweets & missing text,
    groups by user, concatenates all tweets into one doc per user.
    Returns a DataFrame with columns: [author, text].
    """
    df = pd.read_csv(csv_path)
    # drop retweets
    df = df[df["retweet_author"].isnull()]
    # drop missing text/author
    df = df.dropna(subset=["text", "author"])
    # Add party info for later
    df = df.rename(columns={"Party": "party"})
    
    # Clean tweets
    df["text_clean"] = df["text"].apply(clean_text)
    
    # group by author
    user_docs = df.groupby("author").agg({
        "text_clean": " ".join,
        "party": lambda x: x.iloc[0]  # Take the first party label (should be consistent)
    }).reset_index()
    
    return user_docs

# Function to tokenize with expanded stopwords
def tokenize_documents(user_docs):
    """
    Tokenizes each user doc with more comprehensive stopword removal.
    Returns a list of TaggedDocument objects.
    """
    # Expanded stopwords - NLTK + custom
    stop_words = set(stopwords.words("english"))
    # Add custom stopwords
    custom_stops = {"amp", "rt", "via", "im", "thats", "id", "us", "ive", "dont", 
                   "cant", "youre", "youve", "isnt", "wasnt", "didnt", "wont",
                   "couldnt", "shouldnt", "wouldnt", "arent"}
    stop_words.update(custom_stops)
    
    # Add single letters (though already removed in cleaning)
    stop_words.update(set(string.ascii_lowercase))
    
    documents = []
    for row in user_docs.itertuples():
        user = row.author
        party = row.party
        
        # Use simple_preprocess to remove punctuation, short tokens, etc.
        tokens = simple_preprocess(row.text_clean, deacc=True, min_len=3)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        
        if tokens:  # Only add if there are tokens left
            documents.append(TaggedDocument(words=tokens, tags=[user, party]))
    
    return documents
')

# =====================================================
#  STEP 3: Define Python functions for Doc2Vec modeling
# =====================================================

py_run_string('
# Function to train PV-DM model (similar to the original paper)
def train_pvdm_model(tagged_docs, vector_size=200, window=20, min_count=5, epochs=20):
    """
    Trains a Doc2Vec model with PV-DM (dm=1).
    Uses parameters similar to the original paper.
    """
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        dm=1,            # PV-DM approach (as used in original paper)
        dm_concat=False, # average context vectors
        epochs=epochs
    )
    return model
')

# =====================================================
#  STEP 4: Load and process tweets
# =====================================================

py_run_string('
# Load tweets - adjust path as needed
csv_path = "tweets_congress.csv"
user_docs = load_and_preprocess_tweets(csv_path)

# Print some stats about the data
print(f"Loaded data for {len(user_docs)} unique users")
print(f"Party distribution: {user_docs[\'party\'].value_counts().to_dict()}")

# Sample of processed text
print("\nSample of processed text:")
print(user_docs.iloc[0][\'text_clean\'][:200] + "...")
')

# =====================================================
#  STEP 5: Create document tags and train Doc2Vec model
# =====================================================

py_run_string('
# Tokenize and tag
tagged_docs = tokenize_documents(user_docs)

# Print sample to verify (optional)
# print(f"Sample document: {tagged_docs[0]}")
# print(f"Total documents: {len(tagged_docs)}")

# Train PV-DM model
model = train_pvdm_model(tagged_docs)

# Save model
model.save("models/twitter_users_pvdm.model")
')

# =====================================================
#  STEP 6: Extract user embeddings
# =====================================================

py_run_string('
# Create mapping of users to parties
user_party_map = user_docs[["author", "party"]].set_index("author").to_dict()["party"]

# Extract user embeddings
user_ids = []
user_parties = []
user_vecs = []

# We need to handle tags correctly because they can be tuples
for tag in model.docvecs.doctags.keys():
    # If it\'s a tuple, get the first element (the user ID)
    if isinstance(tag, tuple):
        user_id = tag[0]
    else:
        user_id = tag
        
    # Skip if we don\'t have party info
    if user_id not in user_party_map:
        continue
        
    user_ids.append(user_id)
    user_parties.append(user_party_map[user_id])
    user_vecs.append(model.docvecs[tag])

# Create dataframe with embeddings
df_user_vectors = pd.DataFrame(user_vecs)
df_user_vectors["user"] = user_ids
df_user_vectors["party"] = user_parties

# Save to CSV for R analysis
df_user_vectors.to_csv("data/twitter_user_vectors_pvdm.csv", index=False)
')

# =====================================================
#  STEP 7: PCA and visualization in R
# =====================================================

# Load the user-level vectors
user_vecs <- read_csv("data/twitter_user_vectors_pvdm.csv")

# Make sure party is treated as a factor
user_vecs$party <- factor(user_vecs$party, levels = c("D", "R"))

# Extract user and party info
users <- user_vecs$user
parties <- user_vecs$party

# Check data dimensions
cat("Loaded embeddings for", nrow(user_vecs), "users\n")
cat("Party distribution:\n")
print(table(user_vecs$party))

# Remove non-numeric columns for PCA
mat <- user_vecs %>%
  select(-user, -party) %>%
  as.matrix()

# Run PCA
pca <- prcomp(mat, scale. = TRUE)

# Create dataframe for plotting
pca_scores <- as.data.frame(pca$x) %>%
  mutate(
    user = users,
    party = parties
  )

# Calculate variance explained
var_explained <- summary(pca)$importance[2, 1:5] * 100

# Print variance explained
cat("Variance explained by first 5 components:\n")
for (i in 1:5) {
  cat(sprintf("PC%d: %.2f%%\n", i, var_explained[i]))
}

# =====================================================
#  STEP 8: Create visualizations
# =====================================================

# PC1 vs PC2 plot
pc1_pc2_plot <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = party)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("D" = "blue", "R" = "red")) +
  theme_minimal() +
  labs(
    title = "User Embeddings from PV-DM Doc2Vec (Twitter)",
    subtitle = "PC1 vs. PC2",
    x = "PC1 (Potential Ideological Dimension)",
    y = "PC2",
    color = "Party"
  )

# Save the basic plot
ggsave("output/doc2vec_pca_plot.png", pc1_pc2_plot, width = 10, height = 8)

# Identify extreme users on PC1
top_users <- pca_scores %>%
  arrange(desc(PC1)) %>%
  slice_head(n = 5)

bottom_users <- pca_scores %>%
  arrange(PC1) %>%
  slice_head(n = 5)

extreme_users <- bind_rows(top_users, bottom_users)

# Plot with labeled extremes
extreme_users_plot <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = party)) +
  geom_point(alpha = 0.5) +
  geom_text_repel(
    data = extreme_users, 
    aes(label = user),
    size = 3,
    max.overlaps = 15
  ) +
  scale_color_manual(values = c("D" = "blue", "R" = "red")) +
  theme_minimal() +
  labs(title = "Extreme Users on PC1 Dimension")

# Save the extreme users plot
ggsave("output/doc2vec_extreme_users.png", extreme_users_plot, width = 10, height = 8)

# =====================================================
#  STEP 9: Analyze extreme words on PC1
# =====================================================

# Extract PC1 loadings and write to CSV for Python
pc1 <- pca$rotation[, 1]
pc1 <- pc1 / sqrt(sum(pc1^2))
write.csv(pc1, "data/PC1_vector.csv")

# Extract PC2 loadings and write to CSV for Python
pc2 <- pca$rotation[, 2]
pc2 <- pc2 / sqrt(sum(pc2^2))
write.csv(pc2, "data/PC2_vector.csv")

# Project words onto principal components in Python
py_run_string('
# Load the model
from gensim.models import Doc2Vec
model = Doc2Vec.load("models/twitter_users_pvdm.model")

# Function to find extreme words on a principal component
def analyze_extreme_words(pc_file, output_file, component_name="PC1"):
    # Load PC vector
    pc_df = pd.read_csv(pc_file)
    pc = pc_df.iloc[:, 1].values  # take the values column
    
    # Normalize PC vector
    pc = pc / np.linalg.norm(pc)
    
    # Project words onto PC
    word_projections = {}
    for word in model.wv.vocab:
        vec = model.wv[word]
        # Skip rare words with strange vectors
        if np.any(np.isnan(vec)) or np.all(vec == 0):
            continue
            
        # Normalize word vector
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:  # Avoid division by zero
            norm_vec = vec / vec_norm
            # Calculate projection (dot product)
            score = np.dot(pc, norm_vec)
            word_projections[word] = score
    
    # Sort words by projection score
    sorted_words = sorted(word_projections.items(), key=lambda x: x[1])
    
    # Get extreme words on both ends
    neg_extreme = sorted_words[:20]
    pos_extreme = sorted_words[-20:][::-1]  # reverse to get highest first
    
    # Print extreme words
    print(f"Extreme negative end of {component_name}:")
    for word, score in neg_extreme:
        print(f"{word:<20s} {score:.4f}")
    
    print(f"\\nExtreme positive end of {component_name}:")
    for word, score in pos_extreme:
        print(f"{word:<20s} {score:.4f}")
    
    # Save as CSV
    extreme_words_df = pd.DataFrame(
        neg_extreme + pos_extreme, 
        columns=["word", "score"]
    )
    extreme_words_df["end"] = ["negative"] * 20 + ["positive"] * 20
    extreme_words_df.to_csv(output_file, index=False)
    
    return neg_extreme, pos_extreme

# Analyze PC1
neg_extreme_pc1, pos_extreme_pc1 = analyze_extreme_words(
    "data/PC1_vector.csv", 
    "data/extreme_words_pc1.csv", 
    "PC1"
)

# Analyze PC2
neg_extreme_pc2, pos_extreme_pc2 = analyze_extreme_words(
    "data/PC2_vector.csv", 
    "data/extreme_words_pc2.csv", 
    "PC2"
)
')

# =====================================================
#  STEP 10: Visualize extreme words
# =====================================================

# Load extreme words data
extreme_words_pc1 <- read_csv("data/extreme_words_pc1.csv")
extreme_words_pc2 <- read_csv("data/extreme_words_pc2.csv")

# Create plot for PC1 extreme words
pc1_words_plot <- ggplot(extreme_words_pc1, aes(x = score, y = reorder(word, score), fill = end)) +
  geom_col() +
  scale_fill_manual(values = c("negative" = "blue", "positive" = "red")) +
  labs(
    title = "Words at Extremes of PC1 Dimension",
    subtitle = "Potentially Representing Constituency Dimensions",
    x = "PC1 Score",
    y = "Word",
    fill = "PC1 End"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

# Create plot for PC2 extreme words
pc2_words_plot <- ggplot(extreme_words_pc2, aes(x = score, y = reorder(word, score), fill = end)) +
  geom_col() +
  scale_fill_manual(values = c("negative" = "blue", "positive" = "red")) +
  labs(
    title = "Words at Extremes of PC2 Dimension",
    subtitle = "Representing Linguistic Style Variation",
    x = "PC2 Score",
    y = "Word",
    fill = "PC2 End"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.major.y = element_blank()
  )

# Save extreme words plots
ggsave("output/doc2vec_pc1_words.png", pc1_words_plot, width = 12, height = 10)
ggsave("output/doc2vec_pc2_words.png", pc2_words_plot, width = 12, height = 10)

# =====================================================
#  STEP 11: Summary and comparison with original paper
# =====================================================

cat("\n-----------------------------------------------------\n")
cat("SUMMARY OF DOC2VEC PV-DM ANALYSIS ON TWITTER DATA\n")
cat("-----------------------------------------------------\n\n")

cat("1. Primary dimensions identified:\n")
cat("   - PC1: Rural/Agricultural (negative) vs. Hispanic/Spanish-speaking (positive)\n")
cat("   - PC2: Spanish formal discourse (negative) vs. English confrontational rhetoric (positive)\n\n")

cat("2. Unlike the original paper's analysis of parliamentary speeches:\n")
cat("   - No clear partisan separation was observed\n")
cat("   - Demographic and linguistic dimensions dominated over ideological ones\n\n")

cat("3. Variance explained by principal components:\n")
for (i in 1:5) {
  cat(sprintf("   - PC%d: %.2f%%\n", i, var_explained[i]))
}