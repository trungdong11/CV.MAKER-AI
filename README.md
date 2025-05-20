# CV Scoring API

A FastAPI-based API for parsing and analyzing CVs using AI.

## Features

- Parse CVs from PDF and DOC/DOCX files
- Extract structured information using Google's Gemini AI
- Rate limiting to prevent abuse
- Swagger UI documentation
- CORS support

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the API

Start the server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`

## API Endpoints

### Parse CV
```
POST /api/v1/cv/parse
```

Upload a CV file (PDF or DOC/DOCX) to get structured information.

**Request:**
- Content-Type: multipart/form-data
- Body: file (PDF or DOC/DOCX)

**Response:**
```json
{
  "summary": "Professional summary",
  "personalDetails": {
    "fullname": "Full Name",
    "phoneNumber": "Phone Number",
    "address": "Address",
    "email": "email@example.com"
  },
  "socials": [
    {
      "icon": "platform",
      "link": "profile_url"
    }
  ],
  "education": [...],
  "languages": [...],
  "skills": [...],
  "works": [...],
  "projects": [...],
  "certification": [...],
  "organization": [...],
  "award": [...],
  "processing_times": {
    "text_extraction": 0.5,
    "parsing": 2.3,
    "total": 2.8
  }
}
```

## Rate Limiting

The API implements rate limiting:
- 100 requests per hour per IP address
- Rate limit headers are included in responses

## Error Handling

The API returns appropriate HTTP status codes and error messages:
- 400: Bad Request (invalid file type, etc.)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

## License

MIT 