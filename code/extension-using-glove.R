# =====================================================
# Replication Script 2: Extension using GloVe Embeddings
# =====================================================

#### This script:
  
#### 1. Loads and preprocesses congressional tweets with enhanced text cleaning that removes single letters and emojis
#### 2. Builds a term co-occurrence matrix using a context window of 10 words
#### 3. Trains a GloVe model with 100 dimensions
#### 4. Creates user-level embeddings by averaging word vectors for each user
#### 5. Performs principal component analysis (PCA) on these embeddings
#### 6. Identifies and visualizes extreme words on both PC1 and PC2
#### 7. Plots users in the PC1-PC2 space with extreme users labeled
#### 8. Provides a comprehensive summary of the analysis results

# =====================================================
#  STEP 0: Load packages & set up environment
# =====================================================
library(tidyverse)
library(text2vec)    # for GloVe embeddings
library(stopwords)    # standard stopword lists
library(SnowballC)    # for stemming (if needed)
library(ggplot2)      # for visualization
library(ggrepel)      # for text labels in plots
library(data.table)   # for efficient data operations

# Set random seed for reproducibility
set.seed(123)

# =====================================================
#  STEP 1: Read Data & Preprocess
# =====================================================
# Load the congressional tweets data
tweets <- read_csv("tweets_congress.csv")

# Filter to major parties
tweets <- tweets %>% 
  filter(Party %in% c("D","R"))

# Drop retweets to avoid artificial inflation of language patterns
tweets <- tweets %>% 
  filter(is.na(retweet_author))

# Define enhanced text cleaning function with better emoji and single-letter handling
clean_tweet <- function(txt) {
  txt %>%
    str_remove_all("http\\S+|www\\S+") %>%  # remove URLs
    str_remove_all("@\\w+") %>%             # remove @handles
    str_to_lower() %>%                      # to lower case
    str_replace_all("[[:punct:]]", " ") %>% # remove punctuation
    str_remove_all("[0-9]+") %>%            # remove digits
    str_remove_all("\\b[a-zA-Z]\\b") %>%    # remove single letters
    str_remove_all("\\p{So}|\\p{Sk}") %>%   # remove emojis & symbols
    str_squish()                            # trim extra whitespace
}

# Apply cleaning function to tweets
tweets <- tweets %>%
  mutate(text_clean = clean_tweet(text))

# Define expanded stopwords
# Basic English stopwords
basic_stops <- stopwords("en")

# Common Twitter/social media terms
twitter_stops <- c("amp", "rt", "via", "im", "thats", "id", "us", "ive", "dont", 
                   "cant", "youre", "youve", "isnt", "wasnt", "didnt", "wont",
                   "couldnt", "shouldnt", "wouldnt", "arent")

# Single letters (already removed in cleaning, but added here for completeness)
single_letters <- letters

# Combine all stopwords
my_stopwords <- c(basic_stops, twitter_stops, single_letters)


# =====================================================
#  STEP 3: Building the Term Co-occurrence Matrix
# =====================================================

# Tokenize tweets
tokens <- space_tokenizer(tweets$text_clean)

# Create an itoken object with author IDs from the cleaned tweets
it <- itoken(tokens, 
             ids = tweets$author,
             progressbar = TRUE)

# Build vocabulary
vocab <- create_vocabulary(it, stopwords = my_stopwords, 
                           ngram = c(1L, 1L))

# Examine vocabulary size (optional)
#vocab_size <- nrow(vocab)
#cat("Initial vocabulary size:", vocab_size, "\n")

# Prune vocabulary to remove rare words
vocab <- prune_vocabulary(vocab, term_count_min = 5)
pruned_size <- nrow(vocab)
cat("Pruned vocabulary size:", pruned_size, "\n")
cat("Removed", vocab_size - pruned_size, "rare terms\n")

# Create vectorizer
vectorizer <- vocab_vectorizer(vocab)

# Create Term Co-occurrence Matrix with window of 10 words
# The 'skip_grams_window' sets the context window. 
# Larger window -> captures broader semantic relatedness
tcm <- create_tcm(it, vectorizer, skip_grams_window = 10)


# =====================================================
#  STEP 4: Fit GloVe Model
# =====================================================

# Configure and train GloVe model with 100 dimensions
glove_dim <- 100
glove_model <- GlobalVectors$new(rank = glove_dim, 
                                 x_max = 10, 
                                 learning_rate = 0.15)

# Train the model with 15 iterations
fit_glove <- glove_model$fit_transform(tcm, n_iter = 15, 
                                       convergence_tol = 0.01)

# text2vec logs the training loss each epoch. 
#  - 'loss' should steadily decrease
#  - 'Success: early stopping' means it converged or no improvement

# Get context component
context <- glove_model$components

# Combine to get full word embeddings
word_vectors <- fit_glove + t(context)
# Each row is a word in the vocab; each col is a dimension (1..100).

# Save the embeddings for future use
saveRDS(word_vectors, file = "data/tweet_word_vectors_glove.rds")

# =====================================================
#  STEP 5: Create User-Level Embeddings
# =====================================================

# Convert to data.table for more efficient processing
dt_tweets <- as.data.table(tweets)

# ---------------------------------------
# (A) Unnest words (tidyverse approach)
# Tokenize by splitting each tweet into words
tweet_words <- dt_tweets[, .(word = unlist(str_split(text_clean, "\\s+"))), 
                         by = .(author, Party)]

# Remove stopwords and short tokens
tweet_words <- tweet_words[!(word %in% my_stopwords) & nchar(word) > 2]

# ---------------------------------------
# (B) Summarize per user
# We'll do user-level. Then each user has an associated Party if we want.

# Create a wordset for each user (unique words used by each author)
user_wordsets <- tweet_words[, .(wordset = list(unique(word))), by = .(author, Party)]

# Initialize a list to store user embeddings
user_embeddings_list <- list()

# For each user, compute the average of their word embeddings
for (i in 1:nrow(user_wordsets)) {
  # Get list of words for this user
  words_i <- unlist(user_wordsets$wordset[i])
  
  # Find which words are in our vocabulary
  valid_words <- intersect(words_i, rownames(word_vectors))
  
  if (length(valid_words) == 0) {
    # No valid words in vocabulary
    emb_vec <- rep(NA, glove_dim)
  } else {
    # Extract embeddings for these words
    emb_mat <- word_vectors[valid_words, , drop = FALSE]
    
    # Average the embeddings
    emb_vec <- colMeans(emb_mat)
  }
  
  # Store in the list
  user_embeddings_list[[i]] <- emb_vec
}

# ---------------------------------------
# (C) Store embeddings in a dataframe

# Convert list to matrix
user_emb_matrix <- do.call(rbind, user_embeddings_list)

# Create dataframe with user info
user_embeds <- data.frame(
  author = user_wordsets$author,
  party = user_wordsets$Party,
  user_emb_matrix
)

# ---------------------------------------
# Save user embeddings
saveRDS(user_embeds, "data/user_embeddings_glove.rds")


# =====================================================
#  STEP 6: Principal Component Analysis
# =====================================================

# Remove rows with missing values
user_embeds_complete <- user_embeds[complete.cases(user_embeds), ]

# Check how many users we have after removing NA values (optional)
#cat("Total users with complete embeddings:", nrow(user_embeds_complete), "\n")
#cat("Users by party:\n")
#print(table(user_embeds_complete$party))

# Extract embedding matrix for PCA
emb_data <- user_embeds_complete %>%
  select(-author, -party)

# Run PCA
pca_model <- prcomp(emb_data, scale. = TRUE)

# Summary of variance explained
pca_summary <- summary(pca_model)
print(pca_summary)

# Extract PC scores
scores <- as.data.frame(pca_model$x)
scores$author <- user_embeds_complete$author
scores$party <- user_embeds_complete$party

# =====================================================
#  STEP 7: Plotting PCA Results
# =====================================================

# Create base plot of PC1 vs PC2
pca_plot <- ggplot(scores, aes(x = PC1, y = PC2, color = party)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("D" = "blue", "R" = "red")) +
  theme_minimal() +
  labs(
    title = "User-level PCA of Twitter Embeddings",
    subtitle = "PC1 vs. PC2",
    x = "PC1 (Possible ideology axis?)",
    y = "PC2 (Other dimension)"
  )

# Display the plot
print(pca_plot)

# Save the plot
ggsave("output/twitter_glove_pca_plot.png", pca_plot, width = 10, height = 8)

# =====================================================
#  STEP 8: Identifying Extreme Words on PC1
# =====================================================
# Get PC1 vector (loadings)
pc1 <- pca_model$rotation[, 1]

# We'll project each word from our vocabulary onto PC1
word_pc1_scores <- list()

for (word in rownames(word_vectors)) {
  # Get the word vector
  word_vec <- word_vectors[word, ]
  
  # Normalize both vectors for cosine similarity calculation
  word_vec_norm <- word_vec / sqrt(sum(word_vec^2))
  pc1_norm <- pc1 / sqrt(sum(pc1^2))
  
  # Calculate projection (dot product)
  projection <- sum(word_vec_norm * pc1_norm)
  
  # Store in our list
  word_pc1_scores[[word]] <- projection
}

# Convert to named vector, then sort
word_pc1_vector <- unlist(word_pc1_scores)
negative_extreme <- sort(word_pc1_vector)[1:20]
positive_extreme <- sort(word_pc1_vector, decreasing = TRUE)[1:20]

# Print extreme words (optional)
#cat("Extreme negative end of PC1:\n")
#print(negative_extreme)

#cat("\nExtreme positive end of PC1:\n")
#print(positive_extreme)

# Create dataframe for visualization
extreme_words_df <- data.frame(
  word = c(names(negative_extreme), names(positive_extreme)),
  score = c(negative_extreme, positive_extreme),
  end = c(rep("negative", 20), rep("positive", 20))
)

# Plot extreme words
extreme_words_plot <- ggplot(extreme_words_df, aes(x = score, y = reorder(word, score), fill = end)) +
  geom_col() +
  scale_fill_manual(values = c("negative" = "blue", "positive" = "red")) +
  theme_minimal() +
  labs(
    title = "Top Words at Extreme Ends of PC1",
    x = "PC1 Score",
    y = "Word",
    fill = "PC1 End"
  )

# Display the extreme words plot
print(extreme_words_plot)

# Save the extreme words plot
ggsave("output/twitter_glove_extreme_words_pc1.png", extreme_words_plot, width = 12, height = 8)

# =====================================================
#  STEP 9: Examine Most Extreme Words on PC2
# =====================================================

# Get PC2 vector (loadings)
pc2 <- pca_model$rotation[, 2]

# We'll project each word from our vocabulary onto PC2
word_pc2_scores <- list()

for (word in rownames(word_vectors)) {
  # Get the word vector
  word_vec <- word_vectors[word, ]
  
  # Normalize both vectors for cosine similarity calculation
  word_vec_norm <- word_vec / sqrt(sum(word_vec^2))
  pc2_norm <- pc2 / sqrt(sum(pc2^2))
  
  # Calculate projection (dot product)
  projection <- sum(word_vec_norm * pc2_norm)
  
  # Store in our list
  word_pc2_scores[[word]] <- projection
}

# Convert to named vector, then sort
word_pc2_vector <- unlist(word_pc2_scores)
negative_extreme_pc2 <- sort(word_pc2_vector)[1:20]
positive_extreme_pc2 <- sort(word_pc2_vector, decreasing = TRUE)[1:20]

# Print extreme words for PC2
cat("\nExtreme negative end of PC2:\n")
print(negative_extreme_pc2)

cat("\nExtreme positive end of PC2:\n")
print(positive_extreme_pc2)

# Create dataframe for visualization of PC2
extreme_words_pc2_df <- data.frame(
  word = c(names(negative_extreme_pc2), names(positive_extreme_pc2)),
  score = c(negative_extreme_pc2, positive_extreme_pc2),
  end = c(rep("negative", 20), rep("positive", 20))
)

# Plot extreme words for PC2
extreme_words_pc2_plot <- ggplot(extreme_words_pc2_df, aes(x = score, y = reorder(word, score), fill = end)) +
  geom_col() +
  scale_fill_manual(values = c("negative" = "blue", "positive" = "red")) +
  theme_minimal() +
  labs(
    title = "Top Words at Extreme Ends of PC2",
    x = "PC2 Score",
    y = "Word",
    fill = "PC2 End"
  )

# Display the extreme words plot for PC2
print(extreme_words_pc2_plot)

# Save the extreme words plot for PC2
ggsave("output/twitter_glove_extreme_words_pc2.png", extreme_words_pc2_plot, width = 12, height = 8)

# =====================================================
#  STEP 10: Identify Extreme Users 
# =====================================================
# Get the 5 most extreme users on each end of PC1
extreme_pc1_top <- scores %>%
  arrange(desc(PC1)) %>%
  head(5)

extreme_pc1_bottom <- scores %>%
  arrange(PC1) %>%
  head(5)

# Combine for plotting
extreme_users <- bind_rows(extreme_pc1_top, extreme_pc1_bottom)

# Create labeled plot
extreme_users_plot <- ggplot(scores, aes(x = PC1, y = PC2, color = party)) +
  geom_point(alpha = 0.4) +
  geom_text_repel(
    data = extreme_users,
    aes(label = author),
    size = 3,
    max.overlaps = 15
  ) +
  scale_color_manual(values = c("D" = "blue", "R" = "red")) +
  theme_minimal() +
  labs(
    title = "Extreme Users on PC1 Dimension",
    x = "PC1 (Potential ideology axis)",
    y = "PC2"
  )

# Display the extreme users plot
print(extreme_users_plot)

# Save the extreme users plot
ggsave("output/twitter_glove_extreme_users.png", extreme_users_plot, width = 10, height = 8)
