#!/usr/bin/env python3
"""
Dependency installation script for AI Recruiting Platform
Creates virtual environment and installs all dependencies
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import time

def run_command(command, description, shell=True):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=shell, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Error: {e.stderr}")
        return False

def get_venv_path():
    """Get virtual environment path"""
    if platform.system() == "Windows":
        return Path("venv/Scripts/python.exe")
    else:
        return Path("venv/bin/python")

def get_venv_pip():
    """Get virtual environment pip path"""
    if platform.system() == "Windows":
        return Path("venv/Scripts/pip.exe")
    else:
        return Path("venv/bin/pip")

def create_virtual_environment():
    """Create virtual environment"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            print("Trying to remove existing virtual environment...")
            if platform.system() == "Windows":
                print("Trying to remove virtual environment...")
                try:
                    subprocess.run("rmdir /s /q venv", shell=True, capture_output=True)
                    time.sleep(2)
                    
                    if venv_path.exists():
                        print("Could not remove virtual environment automatically")
                        print("Please run: python clean_venv.py")
                        print("Or manually delete the 'venv' folder and try again")
                        return False
                except Exception as e:
                    print(f"Error: {e}")
                    print("Please run: python clean_venv.py")
                    return False
            else:
                subprocess.run("rm -rf venv", shell=True, capture_output=True)
        else:
            print("Using existing virtual environment")
            return True
    
    print("Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
        return False
    
    return True

def activate_and_install():
    """Activate virtual environment and install dependencies"""
    venv_python = get_venv_path()
    venv_pip = get_venv_pip()
    
    if not venv_python.exists():
        print("Virtual environment not found")
        return False
    
    print(f"Using virtual environment: {venv_python}")
    
    # Upgrade pip in virtual environment
    if not run_command(f'"{venv_python}" -m pip install --upgrade pip', "Upgrading pip in virtual environment"):
        return False
    
    # Install core dependencies
    print("\nInstalling core dependencies...")
    if not run_command(f'"{venv_pip}" install -r requirements.txt', "Installing requirements"):
        print("Failed to install requirements")
        return False
    
    # Fix bcrypt issue
    print("\nFixing bcrypt compatibility...")
    run_command(f'"{venv_pip}" uninstall bcrypt -y', "Removing old bcrypt")
    run_command(f'"{venv_pip}" install bcrypt==4.0.1', "Installing compatible bcrypt")
    
    # Install spaCy model
    print("\nInstalling spaCy model...")
    if not run_command(f'"{venv_python}" -m spacy download en_core_web_sm', "Installing spaCy model"):
        print("spaCy model installation failed, but continuing...")
    
    return True

def verify_installations():
    """Verify all installations work correctly"""
    print("\nVerifying installations...")
    
    venv_python = get_venv_path()
    
    # Test imports using virtual environment
    test_script = """
import sys
test_imports = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("sqlalchemy", "SQLAlchemy"),
    ("qdrant_client", "Qdrant"),
    ("google.generativeai", "Gemini API"),
    ("streamlit", "Streamlit"),
    ("sentence_transformers", "Sentence Transformers"),
    ("spacy", "spaCy"),
    ("PyPDF2", "PyPDF2"),
    ("docx", "python-docx"),
    ("jwt", "PyJWT"),
    ("passlib", "Passlib"),
    ("bcrypt", "bcrypt")
]

failed_imports = []
for module, name in test_imports:
    try:
        __import__(module)
        print(f"[OK] {name} imported successfully")
    except ImportError as e:
        print(f"[FAIL] {name} import failed: {e}")
        failed_imports.append(name)

if failed_imports:
    print()
    print("[FAIL] Some imports failed: " + ", ".join(failed_imports))
    sys.exit(1)
else:
    print()
    print("[OK] All imports successful!")
"""
    
    # Write test script to temporary file
    with open("test_imports.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    # Run test script in virtual environment
    success = run_command(f'"{venv_python}" test_imports.py', "Testing imports", shell=False)
    
    # Clean up
    if Path("test_imports.py").exists():
        Path("test_imports.py").unlink()
    
    return success

def create_activation_script():
    """Create activation script for easy environment activation"""
    if platform.system() == "Windows":
        script_content = """@echo off
echo AI Recruiting Platform - Virtual Environment
echo =============================================
echo.
echo To activate the virtual environment, run:
echo   venv\\Scripts\\activate
echo.
echo To start the application, run:
echo   python start.py
echo.
pause
"""
        with open("activate_env.bat", "w", encoding="utf-8") as f:
            f.write(script_content)
    else:
        script_content = """#!/bin/bash
echo "AI Recruiting Platform - Virtual Environment"
echo "============================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the application, run:"
echo "  python start.py"
echo ""
"""
        with open("activate_env.sh", "w", encoding="utf-8") as f:
            f.write(script_content)
        # Make executable
        os.chmod("activate_env.sh", 0o755)

def main():
    """Main installation function"""
    print("AI Recruiting Platform - Virtual Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        sys.exit(1)
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    print(f"Platform: {platform.system()}")
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies in virtual environment
    if not activate_and_install():
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Verify installations
    if not verify_installations():
        print("Some dependencies failed to install correctly")
        sys.exit(1)
    
    # Create activation script
    create_activation_script()
    
    print("\nVirtual environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Create a .env file with your configuration")
    print("3. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("4. Setup PostgreSQL database")
    print("5. Run: python start.py")
    
    print("\nQuick start:")
    if platform.system() == "Windows":
        print("   Run: activate_env.bat")
    else:
        print("   Run: ./activate_env.sh")
    
    print("\nFor detailed setup instructions, see README.md")

if __name__ == "__main__":
    main() 