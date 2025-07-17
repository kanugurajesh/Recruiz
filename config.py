"""
Configuration file for AI Recruiting Platform
Copy this to .env and fill in your actual values
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/recruiting_db")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")

# Qdrant Vector Database
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Frontend Configuration
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")

# Optional: External Qdrant Cloud
# QDRANT_URL = "https://your-cluster.qdrant.io"
# QDRANT_API_KEY = "your-qdrant-cloud-api-key"

# Environment variables template for .env file:
ENV_TEMPLATE = """
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/recruiting_db

# Security
SECRET_KEY=your-secret-key-here

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key

# Gemini API
GEMINI_API_KEY=your-gemini-api-key

# Optional: External Qdrant Cloud
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your-qdrant-cloud-api-key
""" 