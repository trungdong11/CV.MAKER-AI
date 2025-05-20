import logging
import time
import os
import re
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file_service import extract_text_from_pdf, extract_text_from_doc
from app.services.gemini_service import parse_cv_with_gemini
from app.schemas.document import DocumentResponse, Social, Language

router = APIRouter()
logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object"""
    try:
        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %Y",
            "%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
                
        # If no format matches, return current date
        logger.warning(f"Could not parse date: {date_str}, using current date")
        return datetime.now()
    except Exception as e:
        logger.error(f"Error parsing date {date_str}: {str(e)}")
        return datetime.now()

def validate_languages(languages: list) -> list:
    """Validate and transform language data"""
    if not languages:
        return []
    
    validated_languages = []
    for lang in languages:
        if not isinstance(lang, dict):
            continue
            
        language = lang.get("language")
        proficiency = lang.get("proficiency")
        
        if not language:
            continue
            
        # Create Language object with default proficiency if needed
        validated_languages.append(
            Language(
                language=language,
                proficiency=proficiency if proficiency else "Not specified"
            )
        )
    
    return validated_languages

def process_date_and_ongoing(date_str: str) -> tuple:
    """Process date string and determine if ongoing"""
    if not date_str or date_str.lower() in ["present", "current", "ongoing"]:
        return None, True
    return parse_date(date_str), False

def convert_to_html(text: str) -> str:
    """Convert plain text to HTML format"""
    if not text:
        return ""
    
    # Replace newlines with <br>
    text = text.replace('\n', '<br>')
    
    # Convert bullet points
    text = re.sub(r'^\s*[-•*]\s*(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap lists in <ul>
    text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul>\g<0></ul>', text)
    
    # Convert bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert italic text
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    return text

def get_social_info(link: str) -> tuple:
    """Get social icon and formatted link from URL"""
    logger.info(f"Getting social info for link: {link}")
    
    if not link:
        logger.warning("Empty link provided")
        return None, None
        
    # Remove protocol and www
    clean_link = re.sub(r'^https?://(www\.)?', '', link.lower())
    logger.info(f"Cleaned link: {clean_link}")
    
    # Map domains to icons
    social_icons = {
        'github.com': 'github',
        'linkedin.com': 'linkedin',
        'facebook.com': 'facebook',
        'twitter.com': 'twitter',
        'instagram.com': 'instagram',
        'youtube.com': 'youtube',
        'medium.com': 'medium',
        'dev.to': 'dev',
        'stackoverflow.com': 'stackoverflow',
        'behance.net': 'behance',
        'dribbble.com': 'dribbble',
        'gitlab.com': 'gitlab',
        'bitbucket.org': 'bitbucket'
    }
    
    # Find matching domain
    for domain, icon in social_icons.items():
        if domain in clean_link:
            logger.info(f"Found matching icon: {icon} for domain: {domain}")
            return icon, link
            
    logger.warning(f"No matching icon found for link: {link}")
    return None, link

def process_socials(socials: list) -> list:
    """Process social links and icons"""
    logger.info(f"Processing socials: {socials}")
    
    if not socials:
        logger.warning("No socials found in input")
        return []
        
    processed_socials = []
    for social in socials:
        logger.info(f"Processing social item: {social}")
        
        if not isinstance(social, dict):
            logger.warning(f"Invalid social item type: {type(social)}")
            continue
            
        # Get link from either link or icon field
        link = social.get("link")
        if not link and "icon" in social:
            link = social.get("icon")
            
        logger.info(f"Found link: {link}")
        
        if not link:
            logger.warning("No link found in social item")
            continue
            
        icon, _ = get_social_info(link)
        logger.info(f"Extracted icon: {icon} for link: {link}")
        
        processed_socials.append(
            Social(
                icon=icon,  # Will be null if no matching icon found
                link=link
            )
        )
    
    logger.info(f"Processed socials result: {processed_socials}")
    return processed_socials

@router.post("/process", response_model=DocumentResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Upload and process a document file (PDF or DOC/DOCX) to extract structured information
    """
    start_time = time.time()
    logger.info("Starting document processing...")
    
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

        # Parse document with Gemini
        parse_start = time.time()
        result = await parse_cv_with_gemini(text)
        parse_time = time.time() - parse_start
        logger.info(f"Document parsing completed in {parse_time:.2f} seconds")
        
        # Convert summary and descriptions to HTML
        result["summary"] = convert_to_html(result.get("summary", ""))
        
        # Process socials
        if "socials" in result:
            result["socials"] = process_socials(result["socials"])
        
        # Process dates and ongoing status
        if result.get("education"):
            for edu in result["education"]:
                edu["startDate"] = parse_date(edu["startDate"])
                edu["endDate"], _ = process_date_and_ongoing(edu.get("endDate"))
                if "description" in edu:
                    edu["description"] = convert_to_html(edu["description"])
                    
        if result.get("works"):
            for work in result["works"]:
                work["startDate"] = parse_date(work["startDate"])
                work["endDate"], work["isCurrentWorking"] = process_date_and_ongoing(work.get("endDate"))
                if "description" in work:
                    work["description"] = convert_to_html(work["description"])
                    
        if result.get("projects"):
            for project in result["projects"]:
                project["startDate"] = parse_date(project["startDate"])
                project["endDate"], project["isOngoing"] = process_date_and_ongoing(project.get("endDate"))
                if "description" in project:
                    project["description"] = convert_to_html(project["description"])
                    
        if result.get("certification"):
            for cert in result["certification"]:
                cert["issuedDate"] = parse_date(cert["issuedDate"])
                
        if result.get("organization"):
            for org in result["organization"]:
                org["startDate"] = parse_date(org["startDate"])
                org["endDate"], _ = process_date_and_ongoing(org.get("endDate"))
                if "description" in org:
                    org["description"] = convert_to_html(org["description"])
                    
        if result.get("award"):
            for award in result["award"]:
                award["issuedDate"] = parse_date(award["issuedDate"])
                if "description" in award:
                    award["description"] = convert_to_html(award["description"])
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return result
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 