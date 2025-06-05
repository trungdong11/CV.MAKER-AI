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
        "personal_details": {{
            "full_name": "Extract full name if present, otherwise null",
            "phone_number": "Extract phone number if present, otherwise null",
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
                "start_date": "Extract start date if present, otherwise null",
                "end_date": "Extract end date if present, otherwise null",
                "school_link": "Extract school URL if present, otherwise null",
                "city": "Extract city if present, otherwise null",
                "gpa": "Extract GPA if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "languages": [
            {{"language": "Extract language name if present, otherwise null", "proficiency": "Extract proficiency level if present, otherwise null"}}
        ],
        "skills": [
            {{"skill_category": "Extract category if present, otherwise null", "list_of_skill": "Extract skills if present, otherwise null"}}
        ],
        "works": [
            {{
                "company_name": "Extract company name if present, otherwise null",
                "isCurrent_working": "Extract current status if present, otherwise null",
                "position": "Extract position if present, otherwise null",
                "location": "Extract location if present, otherwise null",
                "start_date": "Extract start date if present, otherwise null",
                "end_date": "Extract end date if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "projects": [
            {{
                "name": "Extract project name if present, otherwise null",
                "link": "Extract project URL if present, otherwise null",
                "start_date": "Extract start date if present, otherwise null",
                "end_date": "Extract end date if present, otherwise null",
                "is_ongoing": "Extract ongoing status if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "certification": [
            {{
                "certification_name": "Extract certification name if present, otherwise null",
                "issuing_organization": "Extract organization name if present, otherwise null",
                "issued_date": "Extract issue date if present, otherwise null",
                "certification_link": "Extract certification URL if present, otherwise null",
                "credential_id": "Extract credential ID if present, otherwise null"
            }}
        ],
        "organization": [
            {{
                "name": "Extract organization name if present, otherwise null",
                "position": "Extract position if present, otherwise null",
                "address": "Extract address if present, otherwise null",
                "start_date": "Extract start date if present, otherwise null",
                "end_date": "Extract end date if present, otherwise null",
                "description": "Extract description if present, otherwise null"
            }}
        ],
        "award": [
            {{
                "award_title": "Extract award name if present, otherwise null",
                "award_title_link": "Extract award URL if present, otherwise null",
                "issued_by": "Extract issuer if present, otherwise null",
                "issued_date": "Extract issue date if present, otherwise null",
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
      "name_cv": "Extract the full name from the CV, if not found use null",
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

async def convert_json_to_segments(cv_data: dict) -> dict:
    """
    Convert JSON CV data to segments format
    """
    segments = []
    
    # Add Personal Info
    if cv_data.get("personal_details"):
        personal_info = cv_data["personal_details"]
        personal_text = f"Name: {personal_info.get('full_name', '')}\n"
        personal_text += f"Phone: {personal_info.get('phone_number', '')}\n"
        personal_text += f"Address: {personal_info.get('address', '')}\n"
        personal_text += f"Email: {personal_info.get('email', '')}\n"
        if personal_info.get('job_title'):
            personal_text += f"Job Title: {personal_info['job_title']}\n"
        segments.append({"section": "Personal Info", "text": personal_text.strip()})
    
    # Add Summary
    if cv_data.get("summary"):
        segments.append({"section": "Summary", "text": cv_data["summary"]})
    
    # Add Skills
    if cv_data.get("skills"):
        skills_text = ""
        for skill in cv_data["skills"]:
            skills_text += f"{skill.get('skill_category', '')}: {skill.get('list_of_skill', '')}\n"
        segments.append({"section": "Skills", "text": skills_text.strip()})
    
    # Add Work Experience
    if cv_data.get("works"):
        works_text = ""
        for work in cv_data["works"]:
            works_text += f"Company: {work.get('company_name', '')}\n"
            works_text += f"Position: {work.get('position', '')}\n"
            works_text += f"Location: {work.get('location', '')}\n"
            works_text += f"Period: {work.get('start_date', '')} - {'Present' if work.get('is_current_working') else work.get('end_date', '')}\n"
            if work.get('description'):
                works_text += f"Description: {work['description']}\n"
            works_text += "\n"
        segments.append({"section": "Work Experience", "text": works_text.strip()})
    
    # Add Education
    if cv_data.get("education"):
        edu_text = ""
        for edu in cv_data["education"]:
            edu_text += f"Degree: {edu.get('degree', '')}\n"
            edu_text += f"School: {edu.get('school', '')}\n"
            edu_text += f"Location: {edu.get('city', '')}\n"
            edu_text += f"Period: {edu.get('start_date', '')} - {edu.get('end_date', 'Present')}\n"
            if edu.get('gpa'):
                edu_text += f"GPA: {edu['gpa']}\n"
            if edu.get('description'):
                edu_text += f"Description: {edu['description']}\n"
            edu_text += "\n"
        segments.append({"section": "Education", "text": edu_text.strip()})
    
    # Add Projects
    if cv_data.get("projects"):
        projects_text = ""
        for project in cv_data["projects"]:
            projects_text += f"Project: {project.get('project_name', '')}\n"
            if project.get('project_link'):
                projects_text += f"Link: {project['project_link']}\n"
            projects_text += f"Period: {project.get('start_date', '')} - {'Present' if project.get('is_ongoing') else project.get('end_date', '')}\n"
            if project.get('description'):
                projects_text += f"Description: {project['description']}\n"
            projects_text += "\n"
        segments.append({"section": "Projects", "text": projects_text.strip()})
    
    # Add Certifications
    if cv_data.get("certification"):
        cert_text = ""
        for cert in cv_data["certification"]:
            cert_text += f"Certification: {cert.get('certification_name', '')}\n"
            cert_text += f"Issuing Organization: {cert.get('issuing_organization', '')}\n"
            cert_text += f"Date: {cert.get('issued_date', '')}\n"
            if cert.get('credential_id'):
                cert_text += f"Credential ID: {cert['credential_id']}\n"
            cert_text += "\n"
        segments.append({"section": "Certifications", "text": cert_text.strip()})
    
    # Add Languages
    if cv_data.get("languages"):
        lang_text = ""
        for lang in cv_data["languages"]:
            lang_text += f"{lang.get('language', '')}: {lang.get('proficiency', '')}\n"
        segments.append({"section": "Languages", "text": lang_text.strip()})
    
    # Add Organizations
    if cv_data.get("organization"):
        org_text = ""
        for org in cv_data["organization"]:
            org_text += f"Organization: {org.get('name', '')}\n"
            org_text += f"Position: {org.get('position', '')}\n"
            org_text += f"Location: {org.get('address', '')}\n"
            org_text += f"Period: {org.get('start_date', '')} - {org.get('end_date', 'Present')}\n"
            if org.get('description'):
                org_text += f"Description: {org['description']}\n"
            org_text += "\n"
        segments.append({"section": "Organizational & Volunteering", "text": org_text.strip()})
    
    # Add Awards
    if cv_data.get("award"):
        award_text = ""
        for award in cv_data["award"]:
            award_text += f"Award: {award.get('award_title', '')}\n"
            award_text += f"Issued By: {award.get('issued_by', '')}\n"
            award_text += f"Date: {award.get('issued_date', '')}\n"
            if award.get('description'):
                award_text += f"Description: {award['description']}\n"
            award_text += "\n"
        segments.append({"section": "Awards", "text": award_text.strip()})
    
    return {
        "name_cv": cv_data.get("personal_details", {}).get("full_name", "Unknown"),
        "segments": segments
    } 