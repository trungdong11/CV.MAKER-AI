import logging
import json
import asyncio
import google.generativeai as genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODELS

logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

async def call_gemini_with_fallback(prompt: str, max_retries: int = 3) -> str:
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    contents=prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini API")
                return response.text.strip("```json\n").strip("```")
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed with model {model_name}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                continue
    
    if last_error:
        raise Exception(f"All Gemini models failed after {max_retries} retries. Last error: {str(last_error)}")

async def parse_cv_with_gemini(raw_text: str) -> dict:
    prompt = f"""
    You are an expert in resume analysis. Your task is to extract information from the raw CV text below and structure it according to the specified format. IMPORTANT: Only extract information that exists in the raw text. Do not add, modify, or infer any information.

    Return a JSON object with this structure:
    {{
        "summary": "Extract the professional summary/objective if present, otherwise null",
        "personalDetails": {{
            "fullname": "Extract full name if present, otherwise null",
            "phoneNumber": "Extract phone number if present, otherwise null",
            "address": "Extract address if present, otherwise null",
            "email": "Extract email if present, otherwise null"
        }},
        "socials": [
            {{"icon": "Extract platform name if present, otherwise null", "link": "Extract profile URL if present, otherwise null"}}
        ],
        "education": [
            {{
                "degree": "Extract degree name if present, otherwise null",
                "school": "Extract school name if present, otherwise null",
                "startDate": "Extract start date if present, otherwise null",
                "endDate": "Extract end date if present, otherwise null",
                "schoolLink": "Extract school URL if present, otherwise null",
                "city": "Extract city if present, otherwise null",
                "GPA": "Extract GPA if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "languages": [
            {{"language": "Extract language name if present, otherwise null", "proficiency": "Extract proficiency level if present, otherwise null"}}
        ],
        "skills": [
            {{"skillCategory": "Extract category if present, otherwise null", "listOfSkill": "Extract skills if present, otherwise null"}}
        ],
        "works": [
            {{
                "companyName": "Extract company name if present, otherwise null",
                "isCurrentWorking": "Extract current status if present, otherwise null",
                "position": "Extract position if present, otherwise null",
                "location": "Extract location if present, otherwise null",
                "startDate": "Extract start date if present, otherwise null",
                "endDate": "Extract end date if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "projects": [
            {{
                "name": "Extract project name if present, otherwise null",
                "link": "Extract project URL if present, otherwise null",
                "startDate": "Extract start date if present, otherwise null",
                "endDate": "Extract end date if present, otherwise null",
                "isOngoing": "Extract ongoing status if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "certification": [
            {{
                "certificationName": "Extract certification name if present, otherwise null",
                "issuingOrganization": "Extract organization name if present, otherwise null",
                "issuedDate": "Extract issue date if present, otherwise null",
                "certificationLink": "Extract certification URL if present, otherwise null",
                "credentialId": "Extract credential ID if present, otherwise null"
            }}
        ],
        "organization": [
            {{
                "name": "Extract organization name if present, otherwise null",
                "position": "Extract position if present, otherwise null",
                "address": "Extract address if present, otherwise null",
                "startDate": "Extract start date if present, otherwise null",
                "endDate": "Extract end date if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "award": [
            {{
                "awardTitle": "Extract award name if present, otherwise null",
                "awardTitleLink": "Extract award URL if present, otherwise null",
                "issuer": "Extract issuer if present, otherwise null",
                "issuedDate": "Extract issue date if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ]
    }}

    Rules:
    1. ONLY extract information that exists in the raw text
    2. DO NOT add, modify, or infer any information
    3. If a field is not found in the raw text, use null
    4. If a section is not found in the raw text, use empty array []
    5. Keep the original text format and content exactly as it appears
    6. Do not translate or modify any text

    Raw CV text:
    {raw_text}
    """
    try:
        response_text = await call_gemini_with_fallback(prompt)
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Error parsing CV with Gemini: {str(e)}")
        raise Exception(f"Failed to parse CV: {str(e)}")

async def check_grammar_gemini(text: str) -> list:
    if not text.strip():
        return []
        
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
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from grammar check: {response_text}")
            return []
    except Exception as e:
        logger.error(f"Error checking grammar with Gemini: {str(e)}")
        return []

async def get_suggestions_gemini(section: str, text: str) -> list:
    if not text.strip():
        return []
        
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
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from suggestions: {response_text}")
            return []
    except Exception as e:
        logger.error(f"Error getting suggestions with Gemini: {str(e)}")
        return []

async def segment_cv_gemini(raw_text: str) -> dict:
    """
    Segment CV text into predefined sections using Gemini
    """
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
        raise Exception(f"Failed to segment CV: {str(e)}") 