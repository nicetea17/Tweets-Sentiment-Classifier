# Tweets-Sentiment-Classifier

The goal of this model is to classify tweets' sentiment into three classes: "negative", "neutral", "positive".

# Data:
GloVe embeddings were used to obtain vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

File: [**tweets.csv**](tweets.csv)

The dataset contains ~14,800 short Twitter messages annotated with sentiment labels.
 
Columns of focus are the following:
  text — the tweet content
  label — sentiment category (positive, neutral, negative)
  embeddings - GloVe embeddings

# Model:
MLP - Multilayer Perceptron with 5 layers and 10 epochs.

Used ReLU as activation function and added Dropout with 10% chance.

Loss function is CrossEntropyLoss.

Optimizer is Adam.

Refer to (tweets-sentiment-classifier.ipynb) to see the code and scroll to the very end to test a prediction for your own input text. 

# Results:
Max accuracy: 73.19%

Below is the plot for the training and validation accuracies at each epoch. 

<img src="Accuracy Training vs Validation.jpeg" alt="Accuracy Chart" width="600"/>



Below is the plot for the training and validation losses at each epoch.

<img src="Loss Training vs Validation.jpeg" alt="Loss Chart" width="600"/>



