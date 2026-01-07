# 🇧🇩 Bangla NLP Model Comparison Demo

A local web application to compare two Bangla NLP models across multiple tasks:
- **BanglaBERT Baseline** (few-shot fine-tuned)
- **Meta-Learning Hybrid Model** (ProtoNet + InfoNCE)

## 📋 Features

### Supported Tasks:
1. **Sentiment Analysis** - positive, negative, neutral
2. **Topic Classification** - bangladesh, international, sports, entertainment  
3. **Hate Speech Detection** - hate, non-hate

### Tech Stack:
- **Backend**: FastAPI
- **Frontend**: HTML5 + CSS3 (vanilla JavaScript)
- **Models**: PyTorch + Transformers
- **100% Local** - No external API calls

## 📁 Project Structure

```
Website/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── models.py           # Model loading & inference
│   └── config.py           # Configuration
├── models/
│   ├── banglabert/
│   │   ├── sentiment/      # Put BanglaBERT sentiment model here
│   │   ├── topic/          # Put BanglaBERT topic model here
│   │   └── hate_speech/    # Put BanglaBERT hate speech model here
│   └── meta_learning/
│       ├── sentiment/      # Put Meta-Learning sentiment model here
│       ├── topic/          # Put Meta-Learning topic model here
│       └── hate_speech/    # Put Meta-Learning hate speech model here
├── static/
│   ├── index.html          # Frontend UI
│   └── style.css           # Styling
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Your trained models saved in the appropriate directories

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd Website
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Organize Your Models
Place your trained models in the following structure:
```
models/
├── banglabert/
│   ├── sentiment/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── ...
│   ├── topic/
│   └── hate_speech/
└── meta_learning/
    ├── sentiment/
    ├── topic/
    └── hate_speech/
```

Each model directory should contain:
- `config.json` - Model configuration
- `pytorch_model.bin` or `model.safetensors` - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.txt` - Vocabulary file

### Step 5: Run the Application
```bash
cd backend
python app.py
```

The server will start at: **http://localhost:8000**

## 💻 Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Select a task (Sentiment Analysis, Topic Classification, or Hate Speech Detection)
3. Enter Bangla text in the input box
4. Click "Predict" button
5. View side-by-side predictions from both models with confidence scores and probability distributions

### Keyboard Shortcut
- Press **Ctrl+Enter** in the text area to submit prediction

## 🔧 API Endpoints

### `GET /`
Returns the main HTML page

### `POST /predict`
Make predictions using both models

**Request Body:**
```json
{
  "text": "আপনার বাংলা বাক্য এখানে",
  "task": "sentiment"  // or "topic" or "hate_speech"
}
```

**Response:**
```json
{
  "banglabert": {
    "prediction": "positive",
    "confidence": 0.89,
    "probabilities": {
      "positive": 0.89,
      "negative": 0.08,
      "neutral": 0.03
    }
  },
  "meta_learning": {
    "prediction": "positive",
    "confidence": 0.92,
    "probabilities": {
      "positive": 0.92,
      "negative": 0.06,
      "neutral": 0.02
    }
  }
}
```

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "models_loaded": {
    "banglabert": {
      "sentiment": true,
      "topic": true,
      "hate_speech": true
    },
    "meta_learning": {
      "sentiment": true,
      "topic": true,
      "hate_speech": true
    }
  }
}
```

## 🎨 Customization

### Modify Task Labels
Edit `backend/config.py` to change task labels or add new tasks:
```python
TASKS = {
    "sentiment": {
        "name": "Sentiment Analysis",
        "labels": ["positive", "negative", "neutral"],
        ...
    }
}
```

### Adjust Model Loading
If your meta-learning model has a different architecture, modify the `_load_meta_learning_model` method in `backend/models.py`

### UI Customization
Edit `static/style.css` to change colors, fonts, or layout

## 🐛 Troubleshooting

### Models not loading
- Verify model files exist in correct directories
- Check file permissions
- Ensure model format is compatible with transformers library

### CUDA/GPU issues
- The app automatically detects CUDA availability
- For CPU-only: Works automatically
- For GPU: Ensure PyTorch CUDA version matches your GPU drivers

### Port already in use
Change the port in `backend/app.py`:
```python
uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
```

## 📊 Performance Notes

- First prediction may take longer due to model loading
- CPU inference: ~1-2 seconds per prediction
- GPU inference: ~100-200ms per prediction
- Models are loaded once at startup and kept in memory

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

Created for Bangla NLP research and model comparison

---

**Note**: This is a local demo application. For production deployment, consider adding:
- Authentication
- Rate limiting
- Logging
- Error monitoring
- Model versioning
- Database for storing predictions
