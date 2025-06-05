import logging
import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from app.services.file_service import extract_text_from_pdf, extract_text_from_doc, validate_file_type
from app.services.gemini_service import parse_cv_with_gemini, segment_cv_gemini, check_grammar_gemini, get_suggestions_gemini
from app.schemas.cv import CV, LocalCVRequest
import os
import pickle
import numpy as np
from xgboost import XGBRegressor
from typing import List, Dict, Any
import asyncio
import json

router = APIRouter()
logger = logging.getLogger(__name__)

# Định nghĩa max_scores
max_scores = {
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

# Tổng max_scores để tính thang 100
total_max_score = sum(score for score in max_scores.values() if score > 0)

# Định nghĩa đường dẫn models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
MODEL_PATHS = {
    "xgb_model": os.path.join(MODELS_DIR, "scoring_model.json"),
    "vectorizer": os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"),
    "section_encoder": os.path.join(MODELS_DIR, "section_encoder.pkl")
}

# Cache cho models
model_cache = {
    "model": None,
    "vectorizer": None,
    "section_encoder": None,
    "last_loaded": None
}

# Hàm load models với cache
def load_models():
    global model_cache
    
    # Kiểm tra cache
    current_time = time.time()
    if (model_cache["model"] is not None and 
        model_cache["last_loaded"] is not None and 
        current_time - model_cache["last_loaded"] < 3600):  # Cache for 1 hour
        return (
            model_cache["model"],
            model_cache["vectorizer"],
            model_cache["section_encoder"]
        )
    
    # Kiểm tra sự tồn tại của các file model
    missing_files = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing_files.append(f"{name} ({path})")
    
    if missing_files:
        error_msg = f"Missing model files: {', '.join(missing_files)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Model files not found. Please ensure all required model files are present in the models directory."
        )
    
    # Load models mới nếu cache hết hạn
    try:
        # Load XGBoost model
        model = XGBRegressor()
        model.load_model(MODEL_PATHS["xgb_model"])
        
        # Load vectorizer
        with open(MODEL_PATHS["vectorizer"], "rb") as f:
            vectorizer = pickle.load(f)
            
        # Load section encoder
        with open(MODEL_PATHS["section_encoder"], "rb") as f:
            section_encoder = pickle.load(f)
            
        # Cập nhật cache
        model_cache["model"] = model
        model_cache["vectorizer"] = vectorizer
        model_cache["section_encoder"] = section_encoder
        model_cache["last_loaded"] = current_time
        
        logger.info("Successfully loaded all models")
        return model, vectorizer, section_encoder
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models: {str(e)}"
        )

# Hàm xử lý một section
async def process_section(seg: Dict, model, vectorizer, section_encoder):
    section = seg["section"]
    if section not in max_scores:
        logger.warning(f"Invalid section: {section}")
        return {
            "section": section,
            "content_score": 0,
            "final_score": 0,
            "grammar_errors": {"minor": 0, "severe": 0},
            "grammar_errors_detailed": [],
            "suggestions": [],
            "quality": "Missing"
        }

    # Trích xuất đặc trưng
    feature_start = time.time()
    text_features = vectorizer.transform([seg["text"]]).toarray()
    section_features = section_encoder.transform([[section]])
    features = np.hstack([text_features, section_features])
    feature_time = time.time() - feature_start
    logger.info(f"Feature extraction for {section} completed in {feature_time:.2f} seconds")

    # Dự đoán content_score chuẩn hóa [0, 1]
    predict_start = time.time()
    normalized_score = model.predict(features)[0]
    normalized_score = min(max(0, float(normalized_score)), 1.0)
    predict_time = time.time() - predict_start
    logger.info(f"Score prediction for {section} completed in {predict_time:.2f} seconds")

    # Chuyển về thang gốc
    content_score = normalized_score * max_scores[section]

    # Kiểm tra ngữ pháp bằng Gemini
    grammar_start = time.time()
    grammar_errors_detailed = await check_grammar_gemini(seg["text"])
    grammar_time = time.time() - grammar_start
    logger.info(f"Grammar check for {section} completed in {grammar_time:.2f} seconds")

    minor = sum(1 for e in grammar_errors_detailed if e["type"].lower() in ["grammar", "vocabulary"])
    severe = sum(1 for e in grammar_errors_detailed if e["type"].lower() == "syntax")
    penalty = (minor * 0.2) + (severe * 0.5)

    # Tạo gợi ý cải thiện bằng Gemini
    suggestion_start = time.time()
    suggestions = await get_suggestions_gemini(section, seg["text"])
    suggestion_time = time.time() - suggestion_start
    logger.info(f"Suggestions generation for {section} completed in {suggestion_time:.2f} seconds")

    # Tính final_score
    final_score = max(0, content_score - penalty)

    # Gán nhãn quality
    quality = (
        "Good" if final_score >= max_scores[section] * 0.75
        else "Weak" if final_score > 0
        else "Missing"
    )

    # Log total processing time for this section
    section_total_time = feature_time + predict_time + grammar_time + suggestion_time
    logger.info(f"Total processing time for {section}: {section_total_time:.2f} seconds")

    return {
        "section": section,
        "content_score": round(content_score, 2),
        "final_score": round(final_score, 2),
        "grammar_errors": {"minor": minor, "severe": severe},
        "grammar_errors_detailed": grammar_errors_detailed,
        "suggestions": suggestions,
        "quality": quality
    }

# Hàm tính điểm cho các segment
async def score_segments(segments: List[Dict]):
    try:
        logger.info("Loading model and components")
        model_load_start = time.time()
        model, vectorizer, section_encoder = load_models()
        model_load_time = time.time() - model_load_start
        logger.info(f"Models loaded in {model_load_time:.2f} seconds")

        # Xử lý song song các section
        tasks = [process_section(seg, model, vectorizer, section_encoder) for seg in segments]
        processed_sections = await asyncio.gather(*tasks)

        # Tính điểm thang 100
        total_content_score = sum(section["content_score"] for section in processed_sections)
        total_final_score = sum(section["final_score"] for section in processed_sections)
        content_score_100 = (total_content_score / total_max_score) * 100
        final_score_100 = (total_final_score / total_max_score) * 100

        # Tạo báo cáo
        report = {
            "cv_id": segments[0]["cv_id"] if segments and "cv_id" in segments[0] else "CV154-Cloud_Specialist",
            "sections": processed_sections,
            "total_content_score": round(total_content_score, 2),
            "total_final_score": round(total_final_score, 2),
            "content_score_100": round(content_score_100, 2),
            "final_score_100": round(final_score_100, 2),
            "total_grammar_errors": sum(
                section["grammar_errors"]["minor"] + section["grammar_errors"]["severe"]
                for section in processed_sections
            )
        }
        return report
    except Exception as e:
        logger.error(f"Error scoring segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score CV: {str(e)}")

@router.post("/score")
async def score_cv(file: UploadFile = File(...)):
    """
    Upload and score a CV file (PDF or DOC/DOCX)
    """
    start_time = time.time()
    logger.info("Starting CV processing...")
    
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".doc", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and DOC/DOCX files are allowed."
        )

    try:
        # Extract text based on file type
        text_extract_start = time.time()
        if ext == ".pdf":
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_doc(file)
        text_extract_time = time.time() - text_extract_start
        logger.info(f"Text extraction completed in {text_extract_time:.2f} seconds")

        # Segment CV
        segment_start = time.time()
        segments = await segment_cv_gemini(text)
        segment_time = time.time() - segment_start
        logger.info(f"CV segmentation completed in {segment_time:.2f} seconds")
        
        # Score segments
        scoring_start = time.time()
        report = await score_segments(segments["segments"])
        scoring_time = time.time() - scoring_start
        logger.info(f"CV scoring completed in {scoring_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        # Add timing information to the report
        report["processing_times"] = {
            "text_extraction": round(text_extract_time, 2),
            "segmentation": round(segment_time, 2),
            "scoring": round(scoring_time, 2),
            "total": round(total_time, 2)
        }
        
        return report
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/score-local")
async def score_local_cv(cv_data: Dict[str, Any] = Body(...)):
    """
    Score a CV from raw JSON data
    """
    start_time = time.time()
    logger.info("Starting local CV processing...")
    
    try:
        # Convert JSON to text format
        text = json.dumps(cv_data, indent=2)
        
        # Segment CV using Gemini
        segment_start = time.time()
        segments = await segment_cv_gemini(text)
        segment_time = time.time() - segment_start
        logger.info(f"CV segmentation completed in {segment_time:.2f} seconds")
        
        # Score segments
        score_start = time.time()
        report = await score_segments(segments["segments"])
        score_time = time.time() - score_start
        logger.info(f"CV scoring completed in {score_time:.2f} seconds")
        
        # Add CV ID to report if available
        if "id" in cv_data:
            report["cv_id"] = str(cv_data["id"])
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return report
    except Exception as e:
        logger.error(f"Error processing local CV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CV: {str(e)}"
        ) 