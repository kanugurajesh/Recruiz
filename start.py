# start.py
#!/usr/bin/env python3
"""
Startup script for AI Recruiting Platform
Checks dependencies and starts both backend and frontend
"""
import subprocess
import sys
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
import re
import platform

# Load .env file first
load_dotenv()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ is required to run this application.")
        sys.exit(1)
    print(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible.")

def check_virtual_environment():
    """Check if we're running in a virtual environment"""
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print("⚠️  WARNING: You are not in a virtual environment.")
        print("   It is highly recommended to run this project in a virtual environment.")
        print("   Run `python install_dependencies.py` to set one up automatically.")
        time.sleep(3) # Give user time to read

def check_spacy_model():
    """Check if spaCy model is installed, and if not, install it."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' is installed.")
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found.")
        print("   Attempting to download and install now...")
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("✅ spaCy model installed successfully.")
        except subprocess.CalledProcessError:
            print("❌ ERROR: Failed to automatically install spaCy model.")
            print("   Please run this command manually: `python -m spacy download en_core_web_sm`")
            sys.exit(1)

def check_env_file():
    """Check if .env file exists and contains essential keys."""
    if not Path(".env").exists():
        print("❌ ERROR: `.env` file not found.")
        print("   Please copy `config.py` content to a new `.env` file and fill in your values.")
        sys.exit(1)
    
    required_keys = ["DATABASE_URL", "SECRET_KEY", "GEMINI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"❌ ERROR: The following required environment variables are missing in your .env file: {', '.join(missing_keys)}")
        sys.exit(1)
        
    print("✅ .env file is present and configured.")

def check_service(name, url, check_function):
    """Generic service checker."""
    print(f"Checking service: {name}...")
    if check_function(url):
        print(f"✅ {name} is accessible.")
        return True
    else:
        print(f"❌ ERROR: {name} is not accessible at {url}.")
        if name == "Qdrant":
            print("   Please ensure Qdrant is running. E.g., `docker run -p 6333:6333 qdrant/qdrant`")
        elif name == "PostgreSQL":
            print("   Please ensure your PostgreSQL database is running and the DATABASE_URL is correct.")
        return False

def is_qdrant_ok(url):
    try:
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        response = requests.get(f"{url}/collections", headers={"api-key": qdrant_api_key} if qdrant_api_key else None, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def is_postgres_ok(url):
    try:
        import psycopg2
        conn = psycopg2.connect(url)
        conn.close()
        return True
    except Exception:
        return False

def start_process(command, name):
    """Start a subprocess and return it."""
    print(f"🚀 Starting {name}...")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(5) # Give process time to start
    
    # Check if process started
    if process.poll() is not None:
        print(f"❌ ERROR: {name} failed to start. Error:")
        stdout, stderr = process.communicate()
        print(stderr)
        return None
        
    print(f"✅ {name} started successfully.")
    return process

def main():
    """Main startup function"""
    print("=" * 50)
    print("🤖 AI Recruiting Platform - System Startup")
    print("=" * 50)
    
    # --- PRE-FLIGHT CHECKS ---
    check_python_version()
    check_virtual_environment()
    check_env_file()
    check_spacy_model()

    # --- SERVICE CHECKS ---
    print("\n--- Checking Service Availability ---")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    db_url = os.getenv("DATABASE_URL")
    
    if not check_service("Qdrant", qdrant_url, is_qdrant_ok) or \
       not check_service("PostgreSQL", db_url, is_postgres_ok):
        print("\n❌ One or more required services are not available. Aborting startup.")
        sys.exit(1)

    print("\n--- Launching Application ---")
    
    # --- START PROCESSES ---
    backend_command = [sys.executable, "main.py"]
    frontend_command = [sys.executable, "-m", "streamlit", "run", "frontend.py"]
    
    backend_process = start_process(backend_command, "Backend (FastAPI)")
    if not backend_process: sys.exit(1)
    
    frontend_process = start_process(frontend_command, "Frontend (Streamlit)")
    if not frontend_process:
        backend_process.terminate()
        sys.exit(1)
        
    print("\n" + "=" * 50)
    print("🎉 AI Recruiting Platform is RUNNING!")
    print(f"   Frontend URL: http://localhost:8501")
    print(f"   Backend API Docs: http://localhost:8000/docs")
    print("=" * 50)
    print("\nPress Ctrl+C to stop all services.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down... please wait.")
        frontend_process.terminate()
        backend_process.terminate()
        print("✅ Application stopped.")

if __name__ == "__main__":
    main()