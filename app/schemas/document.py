from pydantic import BaseModel, HttpUrl, EmailStr, Field
from typing import List, Optional
from datetime import datetime

class Social(BaseModel):
    icon: str
    link: Optional[str] = None

class Education(BaseModel):
    degree: str
    school: str
    startDate: datetime
    endDate: Optional[datetime] = None
    schoolLink: Optional[str] = None
    city: Optional[str] = None
    GPA: Optional[float] = None
    description: Optional[str] = None

class Language(BaseModel):
    language: str
    proficiency: Optional[str] = None

class Skill(BaseModel):
    skillCategory: str
    listOfSkill: str

class Work(BaseModel):
    companyName: str
    isCurrentWorking: bool = Field(default=False)
    position: str
    location: str
    startDate: datetime
    endDate: Optional[datetime] = None
    description: str

class Project(BaseModel):
    name: str
    link: Optional[str] = None
    startDate: datetime
    endDate: Optional[datetime] = None
    isOngoing: bool = Field(default=False)
    description: str

class Certification(BaseModel):
    certificationName: str
    issuingOrganization: str
    issuedDate: datetime
    certificationLink: Optional[str] = None
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
    awardTitleLink: Optional[str] = None
    issuer: str
    issuedDate: datetime
    description: str

class PersonalDetails(BaseModel):
    fullname: str
    phoneNumber: str
    address: str
    email: EmailStr

class DocumentResponse(BaseModel):
    summary: str
    personalDetails: PersonalDetails
    socials: List[Social]
    education: List[Education]
    languages: List[Language]
    skills: List[Skill]
    works: List[Work]
    projects: List[Project]
    certification: List[Certification]
    organization: List[Organization]
    award: List[Award] 