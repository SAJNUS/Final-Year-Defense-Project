import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model directories
MODELS_DIR = BASE_DIR / "models"
BANGLABERT_DIR = MODELS_DIR / "banglabert"
META_LEARNING_DIR = MODELS_DIR / "meta_learning"

# Task configurations
TASKS = {
    "sentiment": {
        "name": "Sentiment Analysis",
        "labels": ["positive", "negative", "neutral"],
        "banglabert_path": BANGLABERT_DIR / "sentiment",
        "meta_learning_path": META_LEARNING_DIR / "sentiment"
    },
    "topic": {
        "name": "Topic Classification",
        "labels": ["bangladesh", "international", "sports", "entertainment"],
        "banglabert_path": BANGLABERT_DIR / "topic",
        "meta_learning_path": META_LEARNING_DIR / "topic"
    },
    "hate_speech": {
        "name": "Hate Speech Detection",
        "labels": ["hate", "non-hate"],
        "banglabert_path": BANGLABERT_DIR / "hate_speech",
        "meta_learning_path": META_LEARNING_DIR / "hate_speech"
    }
}

# Model loading timeout (seconds)
MODEL_LOAD_TIMEOUT = 60

# Maximum text length
MAX_TEXT_LENGTH = 512
