import logging
import io
import PyPDF2
from docx import Document
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file: UploadFile) -> str:
    """
    Extract text from a PDF file
    """
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

def extract_text_from_doc(file: UploadFile) -> str:
    """
    Extract text from a DOC/DOCX file
    """
    try:
        with io.BytesIO(file.file.read()) as doc_file:
            doc = Document(doc_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOC: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from DOC: {str(e)}")

def validate_file_type(file: UploadFile) -> str:
    """
    Validate file type and return appropriate extension
    """
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['pdf', 'doc', 'docx']:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF and DOC/DOCX files are allowed."
        )
    return ext 