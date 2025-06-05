from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional
from datetime import datetime
from uuid import UUID

class PersonalDetails(BaseModel):
    full_name: str
    phone_number: str
    address: str
    email: EmailStr

class Social(BaseModel):
    icon: str
    link: HttpUrl

class Education(BaseModel):
    degree: str
    school: str
    start_date: datetime
    end_date: datetime
    school_link: Optional[HttpUrl] = None
    city: str
    gpa: Optional[float] = None
    description: Optional[str] = None

class Language(BaseModel):
    language: str
    proficiency: str

class SkillCategory(BaseModel):
    skill_category: str
    list_of_skill: str

class Work(BaseModel):
    company_name: str
    is_current_working: bool
    position: str
    location: str
    start_date: datetime
    end_date: Optional[datetime] = None
    description: str

class Project(BaseModel):
    name: str
    link: Optional[HttpUrl] = None
    start_date: datetime
    end_date: Optional[datetime] = None
    is_ongoing: bool
    description: str

class Certification(BaseModel):
    certification_name: str
    issuing_organization: str
    issued_date: datetime
    certification_link: Optional[HttpUrl] = None
    credential_id: Optional[str] = None

class Organization(BaseModel):
    name: str
    position: str
    address: str
    start_date: datetime
    end_date: Optional[datetime] = None
    description: str

class Award(BaseModel):
    award_title: str
    award_title_link: Optional[HttpUrl] = None
    issued_by: str
    issued_date: datetime
    description: str

class CV(BaseModel):
    summary: str
    personal_details: PersonalDetails
    socials: List[Social]
    education: List[Education]
    languages: List[Language]
    skills: List[SkillCategory]
    works: List[Work]
    projects: List[Project]
    certification: List[Certification]
    organization: List[Organization]
    award: List[Award]

class Segment(BaseModel):
    section: str
    text: str

class CVInput(BaseModel):
    segments: List[Segment]

class LocalCVRequest(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    summary: Optional[str] = None
    personal_details: Optional[PersonalDetails] = None
    socials: Optional[List[Social]] = None
    education: Optional[List[Education]] = None
    award: Optional[List[Award]] = None
    languages: Optional[List[Language]] = None
    skills: Optional[List[SkillCategory]] = None
    works: Optional[List[Work]] = None
    projects: Optional[List[Project]] = None
    certification: Optional[List[Certification]] = None
    publication: Optional[List[dict]] = None
    organization: Optional[List[Organization]] = None 