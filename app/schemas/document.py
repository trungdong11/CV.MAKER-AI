from pydantic import BaseModel, HttpUrl, EmailStr, Field
from typing import List, Optional
from datetime import datetime

class Social(BaseModel):
    icon: Optional[str] = None
    link: Optional[str] = None

class Education(BaseModel):
    degree: Optional[str] = None
    school: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    school_link: Optional[str] = None
    city: Optional[str] = None
    gpa: Optional[float] = None
    description: Optional[str] = None

class Language(BaseModel):
    language: Optional[str] = None
    proficiency: Optional[str] = None

class Skill(BaseModel):
    skill_category: Optional[str] = None
    list_of_skill: Optional[str] = None

class Work(BaseModel):
    company_name: Optional[str] = None
    is_current_working: Optional[bool] = Field(default=False)
    position: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None

class Project(BaseModel):
    name: Optional[str] = None
    link: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_ongoing: Optional[bool] = Field(default=False)
    description: Optional[str] = None

class Certification(BaseModel):
    certification_name: Optional[str] = None
    issuing_organization: Optional[str] = None
    issued_date: Optional[datetime] = None
    certification_link: Optional[str] = None
    credential_id: Optional[str] = None

class Organization(BaseModel):
    name: Optional[str] = None
    position: Optional[str] = None
    address: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: Optional[str] = None

class Award(BaseModel):
    award_title: Optional[str] = None
    award_title_link: Optional[str] = None
    issued_by: Optional[str] = None
    issued_date: Optional[datetime] = None
    description: Optional[str] = None

class PersonalDetails(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    email: Optional[EmailStr] = None

class DocumentResponse(BaseModel):
    summary: Optional[str] = None
    personal_details: Optional[PersonalDetails] = None
    socials: Optional[List[Social]] = None
    education: Optional[List[Education]] = None
    languages: Optional[List[Language]] = None
    skills: Optional[List[Skill]] = None
    works: Optional[List[Work]] = None
    projects: Optional[List[Project]] = None
    certification: Optional[List[Certification]] = None
    organization: Optional[List[Organization]] = None
    award: Optional[List[Award]] = None
