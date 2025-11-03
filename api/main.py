from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import gensim.downloader as api

# --- Model definition ---
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        li = []
        for i in range(len(layer_sizes) - 1):
            li.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                li.append(nn.ReLU())
                li.append(nn.Dropout(0.1))
        self.model = nn.Sequential(*li)

    def forward(self, x):
        return self.model(x)


# --- Load model + word embeddings ---
input_dim = 50
model = MLP([input_dim, 512, 256, 128, 64, 3])
model.load_state_dict(torch.load("api/tweet_sentiment_model.pt", map_location="cpu"))
model.eval()

# fallback: auto-download GloVe if word_vectors.kv isn’t available
try:
    from gensim.models import KeyedVectors
    word_vectors = KeyedVectors.load("api/word_vectors.kv", mmap="r")
except Exception:
    word_vectors = api.load("glove-twitter-50")


# --- Preprocess ---
def preprocess_tweet(tweet: str):
    tweet = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z\s]", "", tweet.lower())
    words = tweet.split()
    valid_words = [w for w in words if w in word_vectors.key_to_index]

    if not valid_words:
        return np.zeros((50,), dtype=np.float32)

    vectors = [word_vectors[w] for w in valid_words]
    return np.mean(vectors, axis=0)


# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:3000", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TweetInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TweetInput):
    vec = preprocess_tweet(data.text)
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    sentiments = ["Negative", "Neutral", "Positive"]
    return {
        "sentiment": sentiments[pred],
        "confidence": round(float(probs[0, pred]), 3)
    }
