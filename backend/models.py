import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Task configurations with correct max_length for each
TASKS = {
    "sentiment": {
        "banglabert_path": MODELS_DIR / "banglabert" / "sentiment",
        "meta_learning_path": MODELS_DIR / "meta_learning" / "sentiment",
        "banglabert_max_length": 128,
        "meta_learning_max_length": 64,
        "meta_projection_dims": (768, 128, 64),  # input, hidden, output
        "meta_has_dropout": False  # NO dropout in sentiment
    },
    "topic": {
        "banglabert_path": MODELS_DIR / "banglabert" / "topic",
        "meta_learning_path": MODELS_DIR / "meta_learning" / "topic",
        "banglabert_max_length": 128,
        "meta_learning_max_length": 128,
        "meta_projection_dims": (768, 256, 128),  # input, hidden, output
        "meta_has_dropout": True  # Has dropout
    },
    "hate_speech": {
        "banglabert_path": MODELS_DIR / "banglabert" / "hate_speech",
        "meta_learning_path": MODELS_DIR / "meta_learning" / "hate_speech",
        "banglabert_max_length": 128,
        "meta_learning_max_length": 64,
        "meta_projection_dims": (768, 128, 64),  # input, hidden, output
        "meta_has_dropout": True  # Has dropout
    }
}


class HybridProtoNetBERT(nn.Module):
    """Custom Meta-Learning model architecture with Prototypical Networks"""
    
    def __init__(self, model_path, projection_dims, has_dropout=True):
        super().__init__()
        # Load base BERT model
        try:
            self.bert = AutoModel.from_pretrained(str(model_path))
        except:
            self.bert = AutoModel.from_pretrained("csebuetnlp/banglabert")
        
        # Projection layers - different for each task
        input_dim, hidden_dim, output_dim = projection_dims
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]
        if has_dropout:
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb, self.projection(cls_emb)
    
    def euclidean_dist(self, x, y):
        """Calculate euclidean distance between query and prototypes"""
        n, m, d = x.size(0), y.size(0), x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)


class ModelManager:
    """Manages loading and inference for both BanglaBERT and Meta-Learning models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.banglabert_models = {}
        self.banglabert_tokenizers = {}
        self.banglabert_labels = {}
        
        self.meta_learning_models = {}
        self.meta_learning_tokenizers = {}
        self.meta_learning_labels = {}
        self.meta_learning_prototypes = {}  # Store prototypes for meta-learning
        
        self.models_loaded = {
            "banglabert": {},
            "meta_learning": {}
        }
        
        # Load all models
        self._load_all_models()
    
    def _load_label_map(self, model_path: Path) -> dict:
        """Load label_map.json from model directory"""
        label_map_path = model_path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                # Convert to list ordered by index
                return [label_map[str(i)] for i in sorted(int(k) for k in label_map.keys())]
        return []
    
    def _load_all_models(self):
        """Load all models for all tasks"""
        for task_name, task_config in TASKS.items():
            logger.info(f"Loading models for task: {task_name}")
            
            # Load BanglaBERT model
            try:
                self._load_banglabert_model(task_name, task_config)
                self.models_loaded["banglabert"][task_name] = True
                logger.info(f"✓ BanglaBERT model loaded for {task_name}")
            except Exception as e:
                logger.warning(f"✗ Failed to load BanglaBERT model for {task_name}: {e}")
                self.models_loaded["banglabert"][task_name] = False
            
            # Load Meta-Learning model
            try:
                self._load_meta_learning_model(task_name, task_config)
                self.models_loaded["meta_learning"][task_name] = True
                logger.info(f"✓ Meta-Learning model loaded for {task_name}")
            except Exception as e:
                logger.warning(f"✗ Failed to load Meta-Learning model for {task_name}: {e}")
                self.models_loaded["meta_learning"][task_name] = False
    
    def _load_banglabert_model(self, task_name: str, task_config: dict):
        """Load BanglaBERT model for a specific task"""
        model_path = task_config["banglabert_path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load label map
        labels = self._load_label_map(model_path)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model.to(self.device)
        model.eval()
        
        self.banglabert_tokenizers[task_name] = tokenizer
        self.banglabert_models[task_name] = model
        self.banglabert_labels[task_name] = labels
    
    def _load_meta_learning_model(self, task_name: str, task_config: dict):
        """Load Meta-Learning model with custom HybridProtoNetBERT architecture"""
        model_path = task_config["meta_learning_path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load label map
        labels = self._load_label_map(model_path)
        
        # Get projection dimensions for this task
        projection_dims = task_config["meta_projection_dims"]
        has_dropout = task_config.get("meta_has_dropout", True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Load custom model architecture
        model = HybridProtoNetBERT(str(model_path), projection_dims, has_dropout)
        
        # Load model weights
        model_weights_path = model_path / "pytorch_model.bin"
        if model_weights_path.exists():
            model.load_state_dict(torch.load(str(model_weights_path), map_location=self.device))
        else:
            raise FileNotFoundError(f"pytorch_model.bin not found in {model_path}")
        
        model.to(self.device)
        model.eval()
        
        # Load global prototypes
        proto_path = model_path / "global_prototypes.pt"
        if proto_path.exists():
            prototypes = torch.load(str(proto_path), map_location=self.device)
        else:
            raise FileNotFoundError(f"global_prototypes.pt not found in {model_path}")
        
        self.meta_learning_tokenizers[task_name] = tokenizer
        self.meta_learning_models[task_name] = model
        self.meta_learning_labels[task_name] = labels
        self.meta_learning_prototypes[task_name] = prototypes
    
    def predict_banglabert(self, text: str, task: str) -> dict:
        """
        Make prediction using BanglaBERT model
        
        Args:
            text: Input Bangla text
            task: Task name (sentiment, topic, hate_speech)
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        if task not in self.banglabert_models:
            return {
                "error": f"BanglaBERT model not loaded for task: {task}",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            # Get correct max_length for this task
            max_length = TASKS[task]["banglabert_max_length"]
            
            tokenizer = self.banglabert_tokenizers[task]
            model = self.banglabert_models[task]
            labels = self.banglabert_labels[task]
            
            # Tokenize with correct max_length
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                predicted_idx = torch.argmax(probs).item()
                confidence = probs[predicted_idx].item()
            
            # Create probability dict
            probabilities = {
                label: float(probs[i])
                for i, label in enumerate(labels)
            }
            
            return {
                "prediction": labels[predicted_idx],
                "confidence": float(confidence),
                "probabilities": probabilities,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"BanglaBERT prediction error for {task}: {e}")
            return {
                "error": str(e),
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {}
            }
    
    def predict_meta_learning(self, text: str, task: str) -> dict:
        """
        Make prediction using Meta-Learning model with Prototypical Networks
        
        Args:
            text: Input Bangla text
            task: Task name (sentiment, topic, hate_speech)
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        if task not in self.meta_learning_models:
            return {
                "error": f"Meta-Learning model not loaded for task: {task}",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {}
            }
        
        try:
            # Get correct max_length for this task
            max_length = TASKS[task]["meta_learning_max_length"]
            
            tokenizer = self.meta_learning_tokenizers[task]
            model = self.meta_learning_models[task]
            labels = self.meta_learning_labels[task]
            prototypes = self.meta_learning_prototypes[task]
            
            # Tokenize with correct max_length
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Predict using prototypical networks
            with torch.no_grad():
                # Get query embedding
                query_emb, _ = model(inputs["input_ids"], inputs["attention_mask"])
                
                # Calculate distances to prototypes
                dists = model.euclidean_dist(query_emb, prototypes)
                
                # Get prediction (closest prototype)
                predicted_idx = torch.argmin(dists, dim=1).item()
                
                # Convert distances to probabilities
                # Using temperature-scaled softmax on negative distances
                temperature = 10.0  # Higher temp = softer probabilities
                neg_dists = -dists[0] / temperature
                probs = torch.nn.functional.softmax(neg_dists, dim=0)
                confidence = probs[predicted_idx].item()
            
            # Create probability dict
            probabilities = {
                label: float(probs[i])
                for i, label in enumerate(labels)
            }
            
            return {
                "prediction": labels[predicted_idx],
                "confidence": float(confidence),
                "probabilities": probabilities,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Meta-Learning prediction error for {task}: {e}")
            return {
                "error": str(e),
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {}
            }
