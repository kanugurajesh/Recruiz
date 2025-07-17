# 🤖 AI Recruiting Platform

An advanced AI-powered recruiting platform leveraging Gemini API and Qdrant vector database for intelligent candidate-job matching, resume analysis, and automated ranking.

---

## ✨ Features

### For Candidates
- **AI-Powered Job Recommendations**: Personalized job recommendations based on your skills and experience
- **Resume Upload & Analysis**: Upload PDF/DOCX resumes for automatic skill extraction
- **Advanced Matching**: Semantic similarity and Gemini AI for better job matching
- **Application Tracking**: Track your applications and match scores
- **No Duplication**: Smart deduplication prevents multiple applications to the same job

### For Employers
- **AI Candidate Ranking**: AI-powered candidate rankings with detailed match scores
- **Resume Analysis**: View detailed AI reports for each candidate
- **Advanced Search**: Find candidates using semantic search and vector similarity
- **Application Management**: Manage applications with status updates
- **Candidate Recommendations**: Get AI recommendations for your job postings

### Technical Features
- **Vector Database**: Qdrant for fast semantic search and similarity matching
- **Gemini AI Integration**: Advanced semantic matching using Google's Gemini API
- **Resume Processing**: Automatic text extraction and skill identification
- **Real-time Scoring**: Dynamic match score calculation with multiple factors
- **Deduplication**: Smart application deduplication and updates
- **Admin Dashboard**: Manage users and data as an admin

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL
- Qdrant (local or cloud)
- Gemini API key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd AIAgentHack2025-kanugurajesh

# Setup virtual environment and install dependencies (recommended)
python install_dependencies.py

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Environment Configuration
Create a `.env` file based on the template in `config.py` (or let the startup script create one):
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/recruiting_db

# Security
SECRET_KEY=your-secret-key-here

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key

# Gemini API
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Setup Qdrant
#### Option A: Local Qdrant
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant
```
#### Option B: Qdrant Cloud
1. Sign up at https://cloud.qdrant.io/
2. Create a cluster
3. Get your API key
4. Update `.env` with cloud URL and API key

### 4. Setup PostgreSQL
```bash
# Create database
createdb recruiting_db
# Or using psql
psql -c "CREATE DATABASE recruiting_db;"
```

### 5. Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 6. Start the Application

#### One-Command Startup (Recommended)
```bash
python start.py
```
- This script checks dependencies, environment, and starts both backend (FastAPI) and frontend (Streamlit).
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

#### Manual Startup
- **Backend (FastAPI):**
  ```bash
  python main.py
  # or
  uvicorn main:app --reload
  ```
- **Frontend (Streamlit):**
  ```bash
  streamlit run frontend.py
  ```

---

## 📖 Usage Guide

### For Candidates
1. **Register/Login**: Create an account as a candidate
2. **Create Profile**: Fill in your basic information
3. **Upload Resume**: Upload your PDF/DOCX resume
4. **Get Recommendations**: View AI-powered job recommendations
5. **Apply**: Apply to jobs with one click

### For Employers
1. **Register/Login**: Create an account as an employer
2. **Post Jobs**: Create detailed job postings
3. **View Applications**: See ranked applications with AI scores
4. **AI Recommendations**: Get candidate recommendations for your jobs
5. **Manage Applications**: Update application statuses

### For Admins
- Manage all users (candidates, employers, admins)
- Delete users and cascade-delete related data

---

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login user

### Candidates
- `POST /api/candidate/profile` - Create candidate profile
- `POST /api/candidate/upload-resume` - Upload and process resume
- `GET /api/candidate/job-recommendations` - Get AI job recommendations
- `GET /api/candidate/applications` - List your applications
- `POST /api/applications` - Apply for a job

### Employers
- `POST /api/jobs` - Create job posting
- `GET /api/jobs` - List jobs (your jobs if employer, all active jobs if candidate)
- `GET /api/jobs/{job_id}/applications` - View job applications
- `GET /api/employer/job/{job_id}/candidate-recommendations` - Get AI candidate recommendations
- `PUT /api/applications/{application_id}/status` - Update application status

### Analytics
- `GET /api/analytics/dashboard` - Get dashboard analytics

### Admin
- `GET /api/admin/users` - List all users
- `DELETE /api/admin/users/{user_id}` - Delete a user and all related data

---

## 🏗️ Architecture

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Streamlit    │    │   FastAPI     │    │  PostgreSQL   │
│  Frontend     │◄──►│   Backend     │◄──►│   Database    │
└───────────────┘    └───────────────┘    └───────────────┘
                            │
                            ▼
                   ┌───────────────┐
                   │   Qdrant      │
                   │   Vector DB   │
                   └───────────────┘
                            │
                            ▼
                   ┌───────────────┐
                   │   Gemini API  │
                   │   (Google AI) │
                   └───────────────┘
```

---

## 🧠 AI Features

### Semantic Matching
- Sentence transformers for skill similarity
- Gemini API for advanced semantic understanding
- Qdrant vector database for fast similarity search

### Resume Analysis
- Automatic text extraction from PDF/DOCX
- Skill identification and normalization
- Experience and education extraction
- Contact information parsing

### Match Scoring
- **Skills Match (50%)**: Traditional skill matching with semantic similarity
- **Experience Match (20%)**: Experience level compatibility
- **Gemini AI (30%)**: Advanced semantic understanding of job-candidate fit

### Deduplication Logic
- Prevents duplicate applications
- Updates existing applications with new data
- Recalculates match scores with updated information

---

## 🔒 Security Features
- JWT token authentication
- Password hashing with bcrypt
- Role-based access control (candidate, employer, admin)
- Input validation and sanitization

---

## 📊 Performance
- Vector similarity search for fast matching
- Asynchronous AI report generation
- Efficient database queries with relationships
- Caching of embeddings and match scores

---

## 🛠️ Development

### Adding New Features
1. Update database models in `main.py`
2. Add or update API endpoints
3. Update frontend components in `frontend.py`
4. Test with sample data

### Testing
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest
```

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📝 License
This project is licensed under the MIT License.

---

## 🆘 Support
For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

## 🎯 Roadmap
- [ ] Email notifications
- [ ] Interview scheduling
- [ ] Advanced analytics dashboard
- [ ] Mobile app
- [ ] Multi-language support
- [ ] Integration with job boards
- [ ] Advanced AI features (personality matching, cultural fit)