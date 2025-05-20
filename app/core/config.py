import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_PREFIX = "/api/v1"
API_TITLE = "CV Scoring API"
API_DESCRIPTION = "API for scoring and analyzing CVs using AI"
API_VERSION = "1.0.0"

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    # Add other models as needed
]

# Rate Limiting Configuration
RATE_LIMIT = 100
RATE_WINDOW = 3600

# Model Paths
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATHS = {
    "xgb_model": os.path.join(MODELS_DIR, "scoring_model.json"),
    "vectorizer": os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "section_encoder": os.path.join(MODELS_DIR, "section_encoder.pkl")
}

# Scoring Configuration
MAX_SCORES = {
    "Personal Info": 10,
    "Summary": 10,
    "Skills": 15,
    "Work Experience": 20,
    "Education": 10,
    "Projects": 15,
    "Certifications": 5,
    "Languages": 5,
    "Organizational & Volunteering": 5,
    "Awards": 5,
    "Hobbies & Interests": 0
}

# Calculate total max score
TOTAL_MAX_SCORE = sum(score for score in MAX_SCORES.values() if score > 0) 