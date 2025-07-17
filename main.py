# main.py
# backend/main.py
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime, timedelta
import uuid
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, EmailStr
import pandas as pd
from dotenv import load_dotenv
import jwt
from jwt.exceptions import PyJWTError
from passlib.context import CryptContext
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
# text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from io import BytesIO
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import google.generativeai
import os
from dotenv import load_dotenv
import docx
import PyPDF2

# Load environment variables
load_dotenv()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/recruiting_db")
## FIX: Added pool_pre_ping=True to handle stale database connections during development.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security setup
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# AI/ML setup
sentence_model = None
try:
    # Load sentence transformer for semantic similarity
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ## FIX: Dynamically get embedding dimension from the loaded model.
    EMBEDDING_DIMENSION = sentence_model.get_sentence_embedding_dimension()
    print(f"Sentence transformer loaded with embedding dimension: {EMBEDDING_DIMENSION}")
except Exception as e:
    print(f"Warning: Could not load sentence transformer: {e}")
    EMBEDDING_DIMENSION = 384 # Default if model fails to load

## FIX: The entire Qdrant setup is updated to be persistent and sync at startup.

# Qdrant setup
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_CANDIDATES = "candidates"
QDRANT_COLLECTION_JOBS = "jobs"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# This function ensures collections exist without deleting them.
def initialize_qdrant_collections():
    try:
        collections_response = qdrant_client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if QDRANT_COLLECTION_CANDIDATES not in existing_collections:
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_CANDIDATES,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {QDRANT_COLLECTION_CANDIDATES}")
            
        if QDRANT_COLLECTION_JOBS not in existing_collections:
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_JOBS,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection: {QDRANT_COLLECTION_JOBS}")
            
    except Exception as e:
        print(f"Qdrant collection setup error: {e}")

# This function syncs the Postgres DB with Qdrant at startup
def sync_postgres_to_qdrant():
    print("Syncing PostgreSQL jobs to Qdrant...")
    db = SessionLocal()
    try:
        jobs = db.query(Job).filter(Job.is_active == True).all()
        points_to_upsert = []
        for job in jobs:
            # Only generate and upsert if embedding is missing
            if not job.skills_embedding:
                all_skills = (job.required_skills or []) + (job.preferred_skills or [])
                embedding = skill_matcher.generate_skills_embedding(all_skills)
                if embedding:
                    job.skills_embedding = embedding
                    points_to_upsert.append(
                        PointStruct(id=str(job.id), vector=embedding, payload={"job_id": str(job.id)})
                    )
        if points_to_upsert:
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_JOBS,
                points=points_to_upsert,
                wait=True
            )
            db.commit()
            print(f"Upserted {len(points_to_upsert)} new jobs to Qdrant.")
    finally:
        db.close()

# Initialize Qdrant and run the sync on application startup
# initialize_qdrant_collections()
# sync_postgres_to_qdrant()

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_model = None
if GEMINI_API_KEY:
    try:
        google.generativeai.configure(api_key=GEMINI_API_KEY)
        gemini_model = google.generativeai.GenerativeModel("gemini-2.5-flash")
        print("Gemini API model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not configure Gemini API or load model: {e}")
else:
    print("Warning: GEMINI_API_KEY not found. Gemini AI features will be limited or unavailable.")

# Helper: Store candidate embedding in Qdrant
def store_candidate_embedding(candidate_id: str, embedding: list, payload: dict):
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_CANDIDATES,
        points=[PointStruct(id=candidate_id, vector=embedding, payload=payload)]
    )

# Helper: Store job embedding in Qdrant
def store_job_embedding(job_id: str, embedding: list, payload: dict):
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_JOBS,
        points=[PointStruct(id=job_id, vector=embedding, payload=payload)]
    )

# Helper: Search jobs for candidate embedding
def search_jobs_for_candidate(embedding: list, top_k: int = 10):
    ## FIX: Added check for empty embedding to prevent Qdrant client errors.
    if not isinstance(embedding, list) or len(embedding) == 0:
        return []
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_JOBS,
        query_vector=embedding,
        limit=top_k
    )
    return results

# Helper: Search candidates for job embedding
def search_candidates_for_job(embedding: list, top_k: int = 10):
    ## FIX: Added check for empty embedding to prevent Qdrant client errors.
    if not isinstance(embedding, list) or len(embedding) == 0:
        return []
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_CANDIDATES,
        query_vector=embedding,
        limit=top_k
    )
    return results

# Helper: Use Gemini for advanced semantic matching
def gemini_semantic_match(text1: str, text2: str) -> float:
    if not gemini_model:
        ## FIX: More explicit warning when the model is not available.
        print("Gemini model not initialized. Returning 0.0 for semantic match.")
        return 0.0
    try:
        prompt = f"Rate the semantic fit between the following two texts on a scale from 0 to 1.\nText 1: {text1}\nText 2: {text2}\nRespond with only a single floating-point number. Score: "
        response = gemini_model.generate_content(prompt)
        score_str = response.text.strip()
        score = 0.0
        try:
            score = float(score_str)
        except ValueError:
            print(f"Could not parse Gemini score '{score_str}' to float. Defaulting to 0.0.")
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"Gemini semantic match error: {e}")
        return 0.0

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    user_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    jobs = relationship("Job", back_populates="employer", cascade="all, delete-orphan")
    candidate_profile = relationship("CandidateProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    applications = relationship("Application", back_populates="candidate", cascade="all, delete-orphan")

class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    full_name = Column(String)
    email = Column(String)
    ## FIX: Made optional fields nullable in the database schema.
    phone = Column(String, nullable=True)
    skills = Column(ARRAY(String), nullable=True)
    experience_years = Column(Integer)
    education = Column(Text, nullable=True)
    previous_roles = Column(ARRAY(String), nullable=True)
    resume_text = Column(Text, nullable=True)
    resume_filename = Column(String, nullable=True)
    skills_embedding = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)
    certifications = Column(ARRAY(String), nullable=True)
    languages = Column(ARRAY(String), nullable=True)
    location = Column(String, nullable=True)
    salary_expectation = Column(Integer, nullable=True)
    availability = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="candidate_profile")
    applications = relationship("Application", back_populates="candidate_profile", cascade="all, delete-orphan")

class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    title = Column(String)
    description = Column(Text)
    required_skills = Column(ARRAY(String))
    preferred_skills = Column(ARRAY(String), nullable=True)
    min_experience = Column(Integer)
    ## FIX: Made optional fields nullable in the database schema.
    max_experience = Column(Integer, nullable=True)
    education_requirement = Column(String)
    job_type = Column(String)
    location = Column(String)
    remote_allowed = Column(Boolean, default=False)
    salary_range_min = Column(Integer, nullable=True)
    salary_range_max = Column(Integer, nullable=True)
    company_name = Column(String)
    department = Column(String, nullable=True)
    skills_embedding = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    employer = relationship("User", back_populates="jobs")
    applications = relationship("Application", back_populates="job", cascade="all, delete-orphan")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    candidate_profile_id = Column(UUID(as_uuid=True), ForeignKey("candidate_profiles.id"))
    status = Column(String, default="applied")
    match_score = Column(Float, nullable=True)
    matched_skills = Column(ARRAY(String), nullable=True)
    missing_skills = Column(ARRAY(String), nullable=True)
    skill_match_percentage = Column(Float, nullable=True)
    experience_match_percentage = Column(Float, nullable=True)
    education_match = Column(Boolean, nullable=True)
    location_match = Column(Boolean, nullable=True)
    salary_match = Column(Boolean, nullable=True)
    ai_report = Column(Text, nullable=True)
    cover_letter = Column(Text, nullable=True)
    applied_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    job = relationship("Job", back_populates="applications")
    candidate = relationship("User", back_populates="applications")
    candidate_profile = relationship("CandidateProfile", back_populates="applications")

# Pydantic Models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    user_type: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class JobCreate(BaseModel):
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str] = []
    min_experience: int
    max_experience: Optional[int] = None
    education_requirement: str
    job_type: str
    location: str
    remote_allowed: bool = False
    salary_range_min: Optional[int] = None
    salary_range_max: Optional[int] = None
    company_name: str
    department: Optional[str] = None

class JobResponse(BaseModel):
    id: str
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: int
    max_experience: Optional[int]
    education_requirement: str
    job_type: str
    location: str
    remote_allowed: bool
    salary_range_min: Optional[int]
    salary_range_max: Optional[int]
    company_name: str
    department: Optional[str]
    is_active: bool
    created_at: datetime
    applications_count: int = 0

class CandidateProfileCreate(BaseModel):
    full_name: str
    phone: Optional[str] = None
    experience_years: int
    education: str
    summary: Optional[str] = None
    certifications: List[str] = []
    languages: List[str] = []
    location: Optional[str] = None
    salary_expectation: Optional[int] = None
    availability: Optional[str] = None

class ApplicationCreate(BaseModel):
    job_id: str
    cover_letter: Optional[str] = None

class ApplicationResponse(BaseModel):
    id: str
    job_title: str
    candidate_name: str
    status: str
    match_score: Optional[float]
    skill_match_percentage: Optional[float]
    experience_match_percentage: Optional[float]
    applied_at: datetime
    ai_report: Optional[str] = None

class JobRecommendation(BaseModel):
    job: JobResponse
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendation_reason: str

# Advanced AI/ML Services
class AdvancedSkillMatcher:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.skill_synonyms = {
            'python': ['python', 'py', 'python3', 'python programming', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'ecmascript', 'typescript', 'ts'],
            'react': ['react', 'reactjs', 'react.js', 'redux', 'next.js', 'gatsby'],
            'angular': ['angular', 'angularjs', 'angular2+', 'typescript'],
            'vue': ['vue', 'vuejs', 'vue.js', 'nuxt.js'],
            'java': ['java', 'spring', 'spring boot', 'hibernate', 'maven', 'gradle'],
            'machine learning': ['ml', 'machine learning', 'artificial intelligence', 'ai', 'deep learning', 'neural networks'],
            'data science': ['data science', 'data scientist', 'data analysis', 'analytics', 'statistics', 'pandas', 'numpy'],
            'sql': ['sql', 'mysql', 'postgresql', 'oracle', 'database', 'rdbms', 'nosql', 'mongodb'],
            'aws': ['aws', 'amazon web services', 'cloud computing', 'ec2', 's3', 'lambda', 'cloudformation'],
            'docker': ['docker', 'containerization', 'containers', 'podman'],
            'kubernetes': ['kubernetes', 'k8s', 'container orchestration', 'helm'],
            'devops': ['devops', 'ci/cd', 'jenkins', 'gitlab', 'github actions', 'terraform', 'ansible'],
            'frontend': ['frontend', 'front-end', 'ui', 'ux', 'html', 'css', 'sass', 'less'],
            'backend': ['backend', 'back-end', 'api', 'microservices', 'rest', 'graphql'],
            'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'],
            'testing': ['testing', 'unit testing', 'integration testing', 'selenium', 'jest', 'pytest'],
            'agile': ['agile', 'scrum', 'kanban', 'jira', 'project management']
        }
        
        self.expanded_skills = {}
        for main_skill, synonyms in self.skill_synonyms.items():
            for synonym in synonyms:
                self.expanded_skills[synonym.lower()] = main_skill

    def normalize_and_expand_skills(self, skills: List[str]) -> List[str]:
        normalized = []
        for skill in skills:
            skill_lower = skill.lower().strip()
            if skill_lower in self.expanded_skills:
                normalized.append(self.expanded_skills[skill_lower])
            else:
                normalized.append(skill_lower)
        return list(set(normalized))

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if sentence_model is None:
            print("Sentence transformer model not loaded. Cannot calculate semantic similarity.")
            return 0.0
        try:
            embeddings = sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def calculate_advanced_skill_match(self, candidate_skills: List[str], job_skills: List[str]) -> Dict[str, any]:
        ## FIX: Ensure skill lists are not None before processing.
        norm_candidate = self.normalize_and_expand_skills(candidate_skills or [])
        norm_job = self.normalize_and_expand_skills(job_skills or [])
        
        matched_skills = []
        missing_skills = []
        skill_scores = {}
        
        for job_skill in norm_job:
            best_match_score = 0
            best_match_skill = None
            
            if job_skill in norm_candidate:
                matched_skills.append(job_skill)
                skill_scores[job_skill] = 1.0
                continue
            
            for candidate_skill in norm_candidate:
                similarity = self.calculate_semantic_similarity(job_skill, candidate_skill)
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_skill = candidate_skill
            
            if best_match_score > 0.75:
                matched_skills.append(job_skill)
                skill_scores[job_skill] = best_match_score
            else:
                missing_skills.append(job_skill)
                skill_scores[job_skill] = best_match_score
        
        total_score = sum(skill_scores.values())
        max_possible_score = len(norm_job)
        match_percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return {
            'matched_skills': list(set(matched_skills)),
            'missing_skills': missing_skills,
            'match_percentage': match_percentage,
            'skill_scores': skill_scores
        }

    def generate_skills_embedding(self, skills: List[str]) -> List[float]:
        if sentence_model is None:
            print("Sentence transformer model not loaded. Cannot generate embeddings.")
            return []
        try:
            skills_text = ' '.join(skills or [])
            if not skills_text:
                return []
            embedding = sentence_model.encode(skills_text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating skills embedding: {e}")
            return []

class ResumeProcessor:
    def __init__(self):
        self.skill_matcher = AdvancedSkillMatcher()
        
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                ## FIX: Handle potential None return from extract_text.
                text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_content: bytes) -> str:
        try:
            doc = docx.Document(BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            return ""
    
    def extract_resume_data(self, resume_text: str) -> Dict[str, any]:
        extracted_data = {'skills': [], 'experience_years': 0, 'education': '', 'previous_roles': [], 'certifications': [], 'languages': [], 'email': '', 'phone': ''}
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, resume_text)
        if email_match:
            extracted_data['email'] = email_match.group()
        
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phone_match = re.search(phone_pattern, resume_text)
        if phone_match:
            extracted_data['phone'] = phone_match.group()
        
        all_skills = {synonym for synonyms in self.skill_matcher.skill_synonyms.values() for synonym in synonyms}
        found_skills = set()
        resume_lower = resume_text.lower()
        all_skills_sorted = sorted(list(all_skills), key=len, reverse=True)

        for skill in all_skills_sorted:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_lower):
                found_skills.add(self.skill_matcher.expanded_skills.get(skill, skill))
        
        extracted_data['skills'] = list(found_skills)
        
        experience_patterns = [r'(\d+)\+?\s*years?\s*of\s*experience', r'(\d+)\+?\s*years?\s*experience', r'experience\s*of\s*(\d+)\+?\s*years?', r'(\d+)\+?\s*yrs?\s*experience']
        
        for pattern in experience_patterns:
            match = re.search(pattern, resume_lower)
            if match:
                extracted_data['experience_years'] = int(match.group(1))
                break
        
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'diploma']
        education_sentences = []
        sentences = re.split(r'[.!?\n]', resume_text)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in education_keywords):
                education_sentences.append(sentence.strip())
        
        extracted_data['education'] = '. '.join(education_sentences[:3])
        
        role_keywords = ['developer', 'engineer', 'manager', 'analyst', 'scientist', 'consultant', 'designer', 'architect', 'specialist']
        roles = []
        
        for sentence in sentences:
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', sentence.lower()) for keyword in role_keywords):
                roles.append(sentence.strip())
        
        extracted_data['previous_roles'] = list(set(roles))[:5]
        
        return extracted_data

class AIReportGenerator:
    def __init__(self):
        self.skill_matcher = AdvancedSkillMatcher()
    
    def generate_detailed_report(self, candidate_profile: CandidateProfile, job: Job, application: Application) -> str:
        ## FIX: Handle None for skills and ensure valid experience values.
        skill_match = self.skill_matcher.calculate_advanced_skill_match(candidate_profile.skills or [], job.required_skills or [])
        exp_match_val = min((candidate_profile.experience_years or 0) / max(job.min_experience or 1, 1), 1.0)
        exp_analysis = self._analyze_experience_fit(candidate_profile.experience_years, job.min_experience, job.max_experience)
        edu_analysis = self._analyze_education_fit(candidate_profile.education, job.education_requirement)
        location_analysis = self._analyze_location_fit(candidate_profile.location, job.location, job.remote_allowed)
        salary_analysis = self._analyze_salary_fit(candidate_profile.salary_expectation, job.salary_range_min, job.salary_range_max)
        
        report = f"""# Candidate Assessment Report
## Candidate: {candidate_profile.full_name}
## Position: {job.title}
## Overall Match Score: {application.match_score:.1f}%

### Executive Summary
This candidate shows a {self._get_match_level(application.match_score)} for the {job.title} position at {job.company_name}.

### Skills Analysis (Weight: 60%)
- **Skill Match Score**: {skill_match['match_percentage']:.1f}%
- **Matched Skills**: {', '.join(skill_match['matched_skills'])}
- **Missing Skills**: {', '.join(skill_match['missing_skills'])}

### Experience Analysis (Weight: 25%)
- **Experience Match Score**: {exp_match_val * 100:.1f}%
- **Candidate Experience**: {candidate_profile.experience_years} years
- **Required Experience**: {job.min_experience}-{job.max_experience or 'N/A'} years
**Experience Assessment:** {exp_analysis}

### Final Recommendation
{self._generate_final_recommendation(application.match_score, skill_match, exp_match_val)}
---
*Report generated by AI Recruiting Assistant*"""
        return report.strip()
    
    def _analyze_experience_fit(self, candidate_exp: Optional[int], min_exp: Optional[int], max_exp: Optional[int]) -> str:
        candidate_exp = candidate_exp or 0
        min_exp = min_exp or 0
        if candidate_exp >= min_exp and (max_exp is None or candidate_exp <= max_exp):
            return "✓ Candidate meets experience requirements."
        elif candidate_exp < min_exp:
            return f"⚠ Candidate has {min_exp - candidate_exp} year(s) less experience than required."
        else:
            return f"✓ Candidate exceeds experience requirements. May be overqualified."

    def _analyze_education_fit(self, candidate_edu: Optional[str], required_edu: Optional[str]) -> Dict[str, any]:
        if not required_edu: return {"match": True, "analysis": "Sufficient."}
        if not candidate_edu: return {"match": False, "analysis": "Not specified."}
        return {"match": True, "analysis": "Assumed sufficient."}

    def _analyze_location_fit(self, candidate_loc: Optional[str], job_loc: Optional[str], remote_allowed: bool) -> str:
        if remote_allowed: return "✓ Remote work allowed."
        if not candidate_loc or not job_loc: return "⚠ Location information incomplete."
        if candidate_loc.lower().strip() == job_loc.lower().strip(): return "✓ Location match."
        return "⚠ Location mismatch."

    def _analyze_salary_fit(self, candidate_salary: Optional[int], min_salary: Optional[int], max_salary: Optional[int]) -> str:
        if candidate_salary is None: return "⚠ Candidate salary expectation not specified."
        if min_salary is None and max_salary is None: return "✓ No salary range specified for position."
        if max_salary and candidate_salary > max_salary: return f"⚠ Exceeds budget (${candidate_salary:,} > ${max_salary:,})."
        if min_salary and candidate_salary < min_salary: return f"✓ Below budget (${candidate_salary:,} < ${min_salary:,})."
        return f"✓ Within budget."

    def _get_match_level(self, score: float) -> str:
        if score >= 80: return "strong match"
        elif score >= 60: return "good match"
        else: return "moderate match"

    def _generate_final_recommendation(self, match_score: float, skill_match: Dict, exp_match: float) -> str:
        if match_score >= 80: return "**STRONG YES** - Highly recommended for interview."
        elif match_score >= 60: return "**YES** - Recommended for interview."
        else: return "**MAYBE** - Consider if other candidates are limited."

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        ## FIX: Standardized JWT exception handling.
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Application starting up...")
    initialize_qdrant_collections()
    sync_postgres_to_qdrant()
    print("Startup complete.")
    yield
    # Code to run on shutdown (if needed)
    print("Application shutting down.")

# Initialize FastAPI app
app = FastAPI(title="AI Recruiting Platform", description="Advanced AI-powered recruiting platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

skill_matcher = AdvancedSkillMatcher()
resume_processor = ResumeProcessor()
report_generator = AIReportGenerator()

Base.metadata.create_all(bind=engine)

@app.post("/api/auth/register", status_code=201)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password, full_name=user.full_name, user_type=user.user_type)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer", "user_type": user.user_type}

@app.post("/api/auth/login")
async def login_user(form_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.email).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer", "user_type": user.user_type}

## FIX: All endpoints below are reviewed and corrected for logic and robustness.

@app.get("/api/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_info = {"id": str(current_user.id), "email": current_user.email, "full_name": current_user.full_name, "user_type": current_user.user_type}
    profile_info = None
    if current_user.user_type == "candidate":
        candidate_profile = db.query(CandidateProfile).filter(CandidateProfile.user_id == current_user.id).first()
        if candidate_profile:
            profile_info = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in candidate_profile.__dict__.items() if not k.startswith('_')}
    return {"user": user_info, "profile": profile_info}

@app.post("/api/candidate/profile", status_code=201)
async def create_candidate_profile(profile: CandidateProfileCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "candidate":
        raise HTTPException(status_code=403, detail="Only candidates can create profiles")
    if db.query(CandidateProfile).filter(CandidateProfile.user_id == current_user.id).first():
        raise HTTPException(status_code=400, detail="Profile already exists")
    
    db_profile = CandidateProfile(user_id=current_user.id, email=current_user.email, **profile.dict())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    return db_profile

@app.post("/api/candidate/upload-resume")
async def upload_resume(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "candidate":
        raise HTTPException(status_code=403, detail="Only candidates can upload resumes")
    
    candidate_profile = db.query(CandidateProfile).filter(CandidateProfile.user_id == current_user.id).first()
    if not candidate_profile:
        raise HTTPException(status_code=400, detail="Please create a profile first")

    file_content = await file.read()
    if file.filename.lower().endswith('.pdf'):
        resume_text = resume_processor.extract_text_from_pdf(file_content)
    elif file.filename.lower().endswith('.docx'):
        resume_text = resume_processor.extract_text_from_docx(file_content)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and DOCX are supported.")

    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

    extracted_data = resume_processor.extract_resume_data(resume_text)
    skills_embedding = skill_matcher.generate_skills_embedding(extracted_data['skills'])

    # Update profile
    candidate_profile.skills = extracted_data['skills']
    candidate_profile.resume_text = resume_text
    candidate_profile.resume_filename = file.filename
    candidate_profile.skills_embedding = skills_embedding
    if extracted_data.get('experience_years'): candidate_profile.experience_years = extracted_data['experience_years']

    db.commit()
    
    if skills_embedding:
        store_candidate_embedding(str(candidate_profile.id), skills_embedding, {"candidate_id": str(candidate_profile.id), "full_name": candidate_profile.full_name})
    
    return {"message": "Resume uploaded and processed successfully", "extracted_skills": extracted_data['skills']}

## FIX: The entire get_job_recommendations function is corrected and simplified.
## FIX: The get_job_recommendations function is updated to filter out already-applied-to jobs.
@app.get("/api/candidate/job-recommendations")
async def get_job_recommendations(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "candidate":
        raise HTTPException(status_code=403, detail="Only candidates can get recommendations")
    
    profile = db.query(CandidateProfile).filter(CandidateProfile.user_id == current_user.id).first()
    if not profile or not profile.skills_embedding:
        raise HTTPException(status_code=400, detail="Please upload a resume to get recommendations")

    applied_job_ids = {app.job_id for app in db.query(Application.job_id).filter(Application.candidate_id == current_user.id).all()}

    qdrant_results = search_jobs_for_candidate(profile.skills_embedding, top_k=20)
    recommendations = []
    
    for result in qdrant_results:
        job_id_str = result.payload.get("job_id")
        if not job_id_str:
            continue

        job_id = uuid.UUID(job_id_str)
        if job_id in applied_job_ids:
            continue

        job = db.query(Job).filter(Job.id == job_id, Job.is_active == True).first()
        if job:
            # --- Corrected Score Calculation ---
            
            # 1. Qdrant semantic score (0-1), scaled to 0-100, weighted at 50%
            qdrant_score = (result.score * 100) * 0.50
            
            # 2. Skill match score (already 0-100), weighted at 40%
            skill_match_details = skill_matcher.calculate_advanced_skill_match(profile.skills, job.required_skills)
            skill_score = skill_match_details['match_percentage'] * 0.40
            
            # 3. Experience match score (already 0-100), weighted at 10%
            exp_match_score = min((profile.experience_years or 0) / max(job.min_experience or 1, 1), 1.0) * 100
            exp_score = exp_match_score * 0.10

            # The final score is a correctly weighted average between 0 and 100.
            overall_match = qdrant_score + skill_score + exp_score
            
            if overall_match > 40:
                job_response = JobResponse(
                    id=str(job.id),
                    applications_count=db.query(Application).filter(Application.job_id == job.id).count(),
                    **{c.name: getattr(job, c.name) for c in job.__table__.columns if c.name != 'id'}
                )
                recommendations.append(JobRecommendation(
                    job=job_response,
                    match_score=overall_match,
                    matched_skills=skill_match_details['matched_skills'],
                    missing_skills=skill_match_details['missing_skills'],
                    recommendation_reason=f"Strong semantic and skill-based match found by AI."
                ))
    
    recommendations.sort(key=lambda x: x.match_score, reverse=True)
    return {"recommendations": recommendations[:10]}

@app.post("/api/jobs", status_code=201)
async def create_job(job: JobCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "employer":
        raise HTTPException(status_code=403, detail="Only employers can create jobs")
    
    skills_embedding = skill_matcher.generate_skills_embedding(job.required_skills + job.preferred_skills)
    db_job = Job(employer_id=current_user.id, skills_embedding=skills_embedding, **job.dict())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    if skills_embedding:
        store_job_embedding(str(db_job.id), skills_embedding, {"job_id": str(db_job.id), "title": db_job.title})
    
    return {"message": "Job created successfully", "job_id": str(db_job.id)}

@app.get("/api/jobs")
async def get_jobs(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.user_type == "employer":
        # Employers see their own jobs
        jobs = db.query(Job).filter(Job.employer_id == current_user.id).offset(skip).limit(limit).all()
    else:
        # Candidates see all active jobs
        jobs = db.query(Job).filter(Job.is_active == True).offset(skip).limit(limit).all()
    
    job_responses = []
    for job in jobs:
        # Count applications
        applications_count = db.query(Application).filter(Application.job_id == job.id).count()
        
        ## FIX: Explicitly create the JobResponse and convert the UUID 'id' to a string.
        ## This resolves the Pydantic validation error.
        job_response = JobResponse(
            id=str(job.id), # Convert UUID to string here
            title=job.title,
            description=job.description,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            min_experience=job.min_experience,
            max_experience=job.max_experience,
            education_requirement=job.education_requirement,
            job_type=job.job_type,
            location=job.location,
            remote_allowed=job.remote_allowed,
            salary_range_min=job.salary_range_min,
            salary_range_max=job.salary_range_max,
            company_name=job.company_name,
            department=job.department,
            is_active=job.is_active,
            created_at=job.created_at,
            applications_count=applications_count
        )
        job_responses.append(job_response)
    
    return {"jobs": job_responses}

## FIX: The entire apply_for_job function is corrected to prevent the TypeError.
## FIX: The overall_match calculation is corrected to use decimal weights.
@app.post("/api/applications", status_code=201)
async def apply_for_job(application: ApplicationCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "candidate":
        raise HTTPException(status_code=403, detail="Only candidates can apply")
    
    profile = db.query(CandidateProfile).filter(CandidateProfile.user_id == current_user.id).first()
    job = db.query(Job).filter(Job.id == application.job_id).first()
    
    if not profile or not job: 
        raise HTTPException(status_code=404, detail="Profile or Job not found")
    if not profile.resume_text: 
        raise HTTPException(status_code=400, detail="Please upload a resume first")

    if db.query(Application).filter(Application.job_id == job.id, Application.candidate_id == current_user.id).first():
        raise HTTPException(status_code=400, detail="You have already applied for this job.")

    # --- Corrected Score Calculation ---
    skill_match = skill_matcher.calculate_advanced_skill_match(profile.skills, job.required_skills)
    exp_match = min((profile.experience_years or 0) / max(job.min_experience or 1, 1), 1.0) * 100
    
    # Using decimal weights (0.70 and 0.30) for a proper weighted average.
    overall_match = (skill_match['match_percentage'] * 0.70) + (exp_match * 0.30)

    db_app = Application(
        job_id=job.id, 
        candidate_id=current_user.id, 
        candidate_profile_id=profile.id,
        cover_letter=application.cover_letter,
        match_score=overall_match, 
        skill_match_percentage=skill_match['match_percentage'],
        experience_match_percentage=exp_match, 
        matched_skills=skill_match['matched_skills'],
        missing_skills=skill_match['missing_skills']
    )
    
    db.add(db_app)
    db.commit()
    db.refresh(db_app)

    ai_report = report_generator.generate_detailed_report(profile, job, db_app)
    db_app.ai_report = ai_report
    db.commit()

    return {"message": "Application submitted successfully", "application_id": str(db_app.id)}

## FIX: The entire get_candidate_applications function is corrected to prevent the ValidationError.
@app.get("/api/candidate/applications")
async def get_candidate_applications(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "candidate":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    apps = db.query(Application).filter(Application.candidate_id == current_user.id).order_by(Application.applied_at.desc()).all()
    
    # Explicitly create the list of responses to ensure correct type conversion.
    application_responses = []
    for app in apps:
        if not app.job or not app.candidate:
            continue # Skip if related job or candidate has been deleted.

        response = ApplicationResponse(
            id=str(app.id),  # This converts the UUID to a string
            job_title=app.job.title,
            candidate_name=app.candidate.full_name,
            status=app.status,
            match_score=app.match_score,
            skill_match_percentage=app.skill_match_percentage,
            experience_match_percentage=app.experience_match_percentage,
            applied_at=app.applied_at,
            ai_report=app.ai_report
        )
        application_responses.append(response)
        
    return {"applications": application_responses}

## FIX: The entire get_job_applications function is corrected to prevent the ValidationError.
@app.get("/api/jobs/{job_id}/applications")
async def get_job_applications(job_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job or (current_user.user_type == "employer" and str(job.employer_id) != str(current_user.id)):
        raise HTTPException(status_code=403, detail="Forbidden or job not found")
    
    apps = db.query(Application).filter(Application.job_id == job_id).order_by(Application.match_score.desc()).all()
    
    # Explicitly create the list of responses to ensure correct type conversion.
    application_responses = []
    for app in apps:
        # Check if related data exists to prevent errors
        if not app.job or not app.candidate or not app.candidate_profile:
            continue

        response = ApplicationResponse(
            id=str(app.id),  # This converts the UUID to a string
            job_title=app.job.title,
            candidate_name=app.candidate_profile.full_name,
            status=app.status,
            match_score=app.match_score,
            skill_match_percentage=app.skill_match_percentage,
            experience_match_percentage=app.experience_match_percentage,
            applied_at=app.applied_at,
            ai_report=app.ai_report
        )
        application_responses.append(response)
        
    return {"applications": application_responses}

@app.get("/api/applications/{application_id}")
async def get_application_details(application_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == application_id).first()
    if not app: raise HTTPException(status_code=404, detail="Application not found")
    
    is_employer = current_user.user_type == 'employer' and str(app.job.employer_id) == str(current_user.id)
    is_candidate = current_user.user_type == 'candidate' and str(app.candidate_id) == str(current_user.id)
    if not is_employer and not is_candidate:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    return {"application": app, "candidate_profile": app.candidate_profile, "job": app.job}

@app.put("/api/applications/{application_id}/status")
async def update_application_status(application_id: str, status: str = Form(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    app = db.query(Application).filter(Application.id == application_id).first()
    if not app or str(app.job.employer_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Forbidden or application not found")
    
    app.status = status
    app.updated_at = datetime.utcnow()
    db.commit()
    return {"message": "Status updated successfully"}

## FIX: The overall_match calculation is corrected to produce a proper percentage.
@app.get("/api/employer/job/{job_id}/candidate-recommendations")
async def get_candidate_recommendations_for_job(job_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id, Job.employer_id == current_user.id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found or forbidden")

    if not job.skills_embedding: return {"recommendations": []}

    qdrant_results = search_candidates_for_job(job.skills_embedding, top_k=20)
    recommendations = []

    for result in qdrant_results:
        profile = db.query(CandidateProfile).filter(CandidateProfile.id == result.payload.get("candidate_id")).first()
        if profile and profile.skills:
            
            # --- Corrected Score Calculation ---
            qdrant_score_pct = (result.score * 100)
            skill_match = skill_matcher.calculate_advanced_skill_match(profile.skills, job.required_skills)
            exp_match_pct = min((profile.experience_years or 0) / max(job.min_experience or 1, 1), 1.0) * 100
            gemini_score_pct = gemini_semantic_match(f"{' '.join(profile.skills)}", f"{job.title} {' '.join(job.required_skills)}") * 100
            
            overall_match = (qdrant_score_pct * 0.40) + (skill_match['match_percentage'] * 0.30) + (gemini_score_pct * 0.20) + (exp_match_pct * 0.10)
            
            app = db.query(Application).filter(Application.job_id == job.id, Application.candidate_id == profile.user_id).first()
            recommendations.append({
                "candidate_id": str(profile.id), "candidate_name": profile.full_name, "email": profile.email,
                "experience_years": profile.experience_years, "location": profile.location, "skills": profile.skills,
                "match_score": overall_match, "qdrant_score": qdrant_score_pct,
                "skill_match_percentage": skill_match['match_percentage'], "gemini_score": gemini_score_pct,
                "experience_match_percentage": exp_match_pct, "matched_skills": skill_match['matched_skills'],
                "missing_skills": skill_match['missing_skills'], "has_applied": app is not None,
                "application_status": app.status if app else None, "resume_filename": profile.resume_filename,
                "application_id": str(app.id) if app else None
            })
            
    recommendations.sort(key=lambda x: x["match_score"], reverse=True)
    return {"job_title": job.title, "job_id": str(job.id), "recommendations": recommendations[:10]}

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type == "employer":
        job_ids = [r.id for r in db.query(Job.id).filter(Job.employer_id == current_user.id).all()]
        return {
            "jobs_count": len(job_ids),
            "active_jobs_count": db.query(Job).filter(Job.employer_id == current_user.id, Job.is_active == True).count(),
            "total_applications": db.query(Application).filter(Application.job_id.in_(job_ids)).count() if job_ids else 0
        }
    elif current_user.user_type == "candidate":
        return {"applications_count": db.query(Application).filter(Application.candidate_id == current_user.id).count()}
    else: # Admin
        return {"total_users": db.query(User).count(), "total_jobs": db.query(Job).count(), "total_applications": db.query(Application).count()}

# --- ADMIN ENDPOINTS ---
def remove_qdrant_embedding(collection_name: str, point_id: str):
    try:
        qdrant_client.delete(collection_name=collection_name, points_selector=[point_id])
    except Exception as e:
        print(f"Error removing embedding for point {point_id} from {collection_name}: {e}")

@app.get("/api/admin/users")
async def admin_list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "admin": raise HTTPException(status_code=403, detail="Forbidden")
    users = db.query(User).all()
    return {"users": [{"id": str(u.id), "email": u.email, "full_name": u.full_name, "user_type": u.user_type, "created_at": u.created_at, "is_active": u.is_active} for u in users]}

@app.delete("/api/admin/users/{user_id}", status_code=204)
async def admin_delete_user(user_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user.user_type != "admin": raise HTTPException(status_code=403, detail="Forbidden")
    if str(current_user.id) == user_id: raise HTTPException(status_code=400, detail="Admins cannot delete themselves.")
    
    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if not user_to_delete: raise HTTPException(status_code=404, detail="User not found")

    ## FIX: Deletion logic now relies on SQLAlchemy's cascade delete for simplicity and robustness.
    ## Qdrant points still need manual deletion.
    if user_to_delete.user_type == "candidate" and user_to_delete.candidate_profile:
        remove_qdrant_embedding(QDRANT_COLLECTION_CANDIDATES, str(user_to_delete.candidate_profile.id))
    elif user_to_delete.user_type == "employer":
        jobs = db.query(Job).filter(Job.employer_id == user_to_delete.id).all()
        for job in jobs:
            remove_qdrant_embedding(QDRANT_COLLECTION_JOBS, str(job.id))
    
    db.delete(user_to_delete)
    db.commit()
    return {"message": "User deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)