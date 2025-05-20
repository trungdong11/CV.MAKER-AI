from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional
from datetime import datetime

class PersonalDetails(BaseModel):
    fullname: str
    phoneNumber: str
    address: str
    email: EmailStr

class Social(BaseModel):
    icon: str
    link: HttpUrl

class Education(BaseModel):
    degree: str
    school: str
    startDate: datetime
    endDate: datetime
    schoolLink: Optional[HttpUrl] = None
    city: str
    GPA: Optional[float] = None
    description: Optional[str] = None

class Language(BaseModel):
    language: str
    proficiency: str

class SkillCategory(BaseModel):
    skillCategory: str
    listOfSkill: str

class Work(BaseModel):
    companyName: str
    isCurrentWorking: bool
    position: str
    location: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: str

class Project(BaseModel):
    name: str
    link: Optional[HttpUrl] = None
    startDate: datetime
    endDate: Optional[datetime] = None
    isOngoing: bool
    description: str

class Certification(BaseModel):
    certificationName: str
    issuingOrganization: str
    issuedDate: datetime
    certificationLink: Optional[HttpUrl] = None
    credentialId: Optional[str] = None

class Organization(BaseModel):
    name: str
    position: str
    address: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: str

class Award(BaseModel):
    awardTitle: str
    awardTitleLink: Optional[HttpUrl] = None
    issuer: str
    issuedDate: datetime
    description: str

class CV(BaseModel):
    summary: str
    personalDetails: PersonalDetails
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