from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from models import ModelManager

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize model manager
model_manager = ModelManager()


class PredictionRequest(BaseModel):
    text: str
    task: str


class PredictionResponse(BaseModel):
    banglabert: dict
    meta_learning: dict


@app.get("/")
async def read_root():
    """Serve the frontend HTML"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "models_loaded": model_manager.models_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions using both models
    
    Args:
        text: Bangla text input
        task: One of 'sentiment', 'topic', 'hate_speech'
    """
    if request.task not in ['sentiment', 'topic', 'hate_speech']:
        raise HTTPException(
            status_code=400,
            detail="Invalid task. Must be 'sentiment', 'topic', or 'hate_speech'"
        )
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Get predictions from both models
        banglabert_result = model_manager.predict_banglabert(request.text, request.task)
        meta_learning_result = model_manager.predict_meta_learning(request.text, request.task)
        
        return PredictionResponse(
            banglabert=banglabert_result,
            meta_learning=meta_learning_result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
