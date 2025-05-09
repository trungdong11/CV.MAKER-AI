import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import pickle
import numpy as np
from xgboost import XGBRegressor
from typing import List, Dict, Optional
import uvicorn
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from docx import Document
import json
import io
from functools import lru_cache
from datetime import datetime, timedelta
from collections import defaultdict

# Load environment variables
load_dotenv()

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("cv_scoring_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cấu hình Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Định nghĩa các model Gemini và thứ tự ưu tiên
GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-pro"
]

# Cấu hình Swagger
SWAGGER_UI_PARAMETERS = {
    "title": "CV Scoring API",
    "description": "API for scoring and analyzing CVs using AI",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json"
}

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
total_max_score = sum(score for score in max_scores.values() if score > 0)  # 100

# Định nghĩa schema cho dữ liệu đầu vào
class Segment(BaseModel):
    section: str
    text: str

class CVInput(BaseModel):
    segments: List[Segment]

# Khởi tạo FastAPI app với cấu hình Swagger
app = FastAPI(**SWAGGER_UI_PARAMETERS)

# Rate limiting configuration
RATE_LIMIT = 100  # requests
RATE_WINDOW = 3600  # seconds (1 hour)
request_history = defaultdict(list)

# Custom rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = datetime.now()
    
    # Clean old requests
    request_history[client_ip] = [
        req_time for req_time in request_history[client_ip]
        if now - req_time < timedelta(seconds=RATE_WINDOW)
    ]
    
    # Check rate limit
    if len(request_history[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    # Add current request
    request_history[client_ip].append(now)
    
    # Process request
    response = await call_next(request)
    return response

# Cache cho models
@lru_cache(maxsize=1)
def load_models():
    model = XGBRegressor()
    model.load_model("./models/scoring_model.json")
    with open("./models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("./models/section_encoder.pkl", "rb") as f:
        section_encoder = pickle.load(f)
    return model, vectorizer, section_encoder

# Hàm gọi Gemini API với fallback
async def call_gemini_with_fallback(prompt: str, max_retries: int = 3) -> str:
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip("```json\n").strip("```")
        except Exception as e:
            last_error = e
            logger.warning(f"Failed with model {model_name}: {str(e)}")
            continue
    
    if last_error:
        raise HTTPException(
            status_code=500,
            detail=f"All Gemini models failed. Last error: {str(last_error)}"
        )

# Hàm trích xuất văn bản từ PDF
def extract_text_from_pdf(file: UploadFile):
    try:
        with io.BytesIO(file.file.read()) as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except PyPDF2.PdfReadError as e:
        logger.error(f"Invalid PDF file: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid PDF file format")
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Hàm trích xuất văn bản từ DOC/DOCX
def extract_text_from_doc(file: UploadFile):
    try:
        with io.BytesIO(file.file.read()) as doc_file:
            doc = Document(doc_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOC: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from DOC: {str(e)}")

# Hàm phân đoạn CV bằng Gemini
async def segment_cv_gemini(raw_text: str):
    prompt = f"""
    You are an expert in resume analysis. Parse the following raw CV text and segment it into predefined sections without modifying, translating, or adding any content. Assign each piece of text to the appropriate section based on its content. Return a JSON object with the structure:
    {{
      "cv_id": "CV154-Cloud_Specialist",
      "segments": [
        {{"section": "Personal Info", "text": "..."}},
        {{"section": "Summary", "text": "..."}},
        {{"section": "Skills", "text": "..."}},
        {{"section": "Work Experience", "text": "..."}},
        {{"section": "Education", "text": "..."}},
        {{"section": "Projects", "text": "..."}},
        {{"section": "Certifications", "text": "..."}},
        {{"section": "Languages", "text": "..."}},
        {{"section": "Organizational & Volunteering", "text": "..."}},
        {{"section": "Awards", "text": "..."}},
      ]
    }}
    Ensure the text in each section is exactly as it appears in the raw text, with no changes, translations, or additions. If a section is missing, include it with an empty "text". If a piece of text does not clearly belong to any section, assign it to the most relevant section based on context.

    Raw CV text:
    {raw_text}
    """
    try:
        response_text = await call_gemini_with_fallback(prompt)
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error segmenting CV with Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to segment CV: {str(e)}")

# Hàm kiểm tra ngữ pháp bằng Gemini
async def check_grammar_gemini(text: str):
    prompt = f"""
    You are a professional proofreader. Analyze the following text and identify any grammar, syntax, or word usage errors. For each error, provide:
    - Location (line or segment).
    - Error type (grammar, syntax, vocabulary).
    - Description of the error.
    - Suggested correction.
    Return the results as a list of errors in JSON format:
    [
      {{"location": "...", "type": "...", "description": "...", "suggestion": "..."}},
      ...
    ]
    Ensure the analysis is in English.

    Text:
    {text}
    """
    try:
        response_text = await call_gemini_with_fallback(prompt)
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error checking grammar with Gemini: {str(e)}")
        return []

# Hàm tạo gợi ý cải thiện bằng Gemini
async def get_suggestions_gemini(section: str, text: str):
    prompt = f"""
    You are a career consultant. Analyze the content of a CV section (section: {section}) and provide specific suggestions to improve its quality. Focus on:
    - Clarity and professionalism in expression.
    - Adding critical details (if missing).
    - Structuring content to stand out.
    Provide up to 3 suggestions, each with:
    - Issue description.
    - Suggested improvement.
    Return the results as a list in JSON format:
    [
      {{"issue": "...", "suggestion": "..."}},
      ...
    ]
    Ensure all suggestions are in English.

    Content:
    {text}
    """
    try:
        response_text = await call_gemini_with_fallback(prompt)
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error getting suggestions with Gemini: {str(e)}")
        return []

# Hàm tính điểm cho các segment
async def score_segments(segments: List[Dict]):
    try:
        logger.info("Loading model and components")
        model, vectorizer, section_encoder = load_models()

        for seg in segments:
            section = seg["section"]
            if section not in max_scores:
                logger.warning(f"Invalid section: {section}")
                seg["content_score"] = 0
                seg["final_score"] = 0
                seg["grammar_errors"] = {"minor": 0, "severe": 0}
                seg["grammar_errors_detailed"] = []
                seg["suggestions"] = []
                seg["quality"] = "Missing"
                continue

            # Trích xuất đặc trưng
            text_features = vectorizer.transform([seg["text"]]).toarray()
            section_features = section_encoder.transform([[section]])
            features = np.hstack([text_features, section_features])

            # Dự đoán content_score chuẩn hóa [0, 1]
            normalized_score = model.predict(features)[0]
            normalized_score = min(max(0, float(normalized_score)), 1.0)

            # Chuyển về thang gốc
            content_score = normalized_score * max_scores[section]
            seg["content_score"] = round(content_score, 2)

            # Kiểm tra ngữ pháp bằng Gemini
            grammar_errors_detailed = await check_grammar_gemini(seg["text"])
            minor = sum(1 for e in grammar_errors_detailed if e["type"].lower() in ["grammar", "vocabulary"])
            severe = sum(1 for e in grammar_errors_detailed if e["type"].lower() == "syntax")
            penalty = (minor * 0.2) + (severe * 0.5)
            seg["grammar_errors"] = {"minor": minor, "severe": severe}
            seg["grammar_errors_detailed"] = grammar_errors_detailed

            # Tạo gợi ý cải thiện bằng Gemini
            suggestions = await get_suggestions_gemini(section, seg["text"])
            seg["suggestions"] = suggestions

            # Tính final_score
            final_score = max(0, content_score - penalty)
            seg["final_score"] = round(final_score, 2)

            # Gán nhãn quality
            seg["quality"] = (
                "Good" if final_score >= max_scores[section] * 0.75
                else "Weak" if final_score > 0
                else "Missing"
            )

        # Tính điểm thang 100
        total_content_score = sum(seg["content_score"] for seg in segments)
        total_final_score = sum(seg["final_score"] for seg in segments)
        content_score_100 = (total_content_score / total_max_score) * 100
        final_score_100 = (total_final_score / total_max_score) * 100

        # Tạo báo cáo
        report = {
            "cv_id": segments[0]["cv_id"] if segments and "cv_id" in segments[0] else "CV154-Cloud_Specialist",
            "sections": [
                {
                    "section": seg["section"],
                    "content_score": seg["content_score"],
                    "final_score": seg["final_score"],
                    "grammar_errors": seg["grammar_errors"],
                    "grammar_errors_detailed": seg["grammar_errors_detailed"],
                    "suggestions": seg["suggestions"],
                    "quality": seg["quality"]
                } for seg in segments
            ],
            "total_content_score": round(total_content_score, 2),
            "total_final_score": round(total_final_score, 2),
            "content_score_100": round(content_score_100, 2),
            "final_score_100": round(final_score_100, 2),
            "total_grammar_errors": sum(
                seg["grammar_errors"]["minor"] + seg["grammar_errors"]["severe"]
                for seg in segments
            )
        }
        return report
    except Exception as e:
        logger.error(f"Error scoring segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to score CV: {str(e)}")

# API Endpoints
@app.post("/upload_cv", response_model=dict)
async def upload_cv(file: UploadFile = File(...)):
    """
    Upload and analyze a CV file (PDF or DOC/DOCX)
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".doc", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and DOC/DOCX files are allowed."
        )

    try:
        # Extract text based on file type
        if ext == ".pdf":
            text = extract_text_from_pdf(file)
            print(text, 'check text pdf');
        else:
            text = extract_text_from_doc(file)
            print(text, 'check text doc');

        # Segment CV
        segments = await segment_cv_gemini(text)
        
        # Score segments
        report = await score_segments(segments["segments"])
        
        return report
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=SWAGGER_UI_PARAMETERS["title"],
        version=SWAGGER_UI_PARAMETERS["version"],
        description=SWAGGER_UI_PARAMETERS["description"],
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)