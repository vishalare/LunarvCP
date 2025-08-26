#!/usr/bin/env python3
"""
🚀 Lunarv Deployment Helper Script
Automates common deployment tasks
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'churn_model.py',
        'data_utils.py',
        'visualization_utils.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def setup_git():
    """Initialize git repository if not already done"""
    if Path('.git').exists():
        print("✅ Git repository already exists")
        return True
    
    print("🔄 Setting up Git repository...")
    
    commands = [
        "git init",
        "git add .",
        "git commit -m 'Initial commit: Lunarv Churn Prediction Tool'",
        "git branch -M main"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    
    print("✅ Git repository initialized")
    return True

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud"""
    print("\n🚀 Streamlit Cloud Deployment")
    print("=" * 40)
    
    if not check_requirements():
        return False
    
    if not setup_git():
        return False
    
    print("\n📋 Next Steps:")
    print("1. Push your code to GitHub:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/lunarv.git")
    print("   git push -u origin main")
    print("\n2. Go to https://share.streamlit.io")
    print("3. Sign in with GitHub")
    print("4. Click 'New app'")
    print("5. Select your repository")
    print("6. Set main file path: app.py")
    print("7. Click 'Deploy!'")
    
    return True

def deploy_heroku():
    """Deploy to Heroku"""
    print("\n☁️ Heroku Deployment")
    print("=" * 40)
    
    if not check_requirements():
        return False
    
    if not setup_git():
        return False
    
    # Check if Heroku CLI is installed
    heroku_check = run_command("heroku --version", "Checking Heroku CLI")
    if not heroku_check:
        print("❌ Heroku CLI not found. Please install it first:")
        print("   Windows: winget install --id=Heroku.HerokuCLI")
        print("   macOS: brew tap heroku/brew && brew install heroku")
        print("   Linux: curl https://cli-assets.heroku.com/install.sh | sh")
        return False
    
    print("\n📋 Next Steps:")
    print("1. Login to Heroku:")
    print("   heroku login")
    print("\n2. Create Heroku app:")
    print("   heroku create lunarv-churn-prediction")
    print("\n3. Add buildpack:")
    print("   heroku buildpacks:set heroku/python")
    print("\n4. Deploy:")
    print("   git push heroku main")
    print("\n5. Open app:")
    print("   heroku open")
    
    return True

def deploy_docker():
    """Deploy using Docker"""
    print("\n🐳 Docker Deployment")
    print("=" * 40)
    
    if not check_requirements():
        return False
    
    # Check if Docker is installed
    docker_check = run_command("docker --version", "Checking Docker")
    if not docker_check:
        print("❌ Docker not found. Please install Docker Desktop first.")
        return False
    
    print("\n📋 Building and running with Docker...")
    
    # Build Docker image
    if not run_command("docker build -t lunarv .", "Building Docker image"):
        return False
    
    print("\n✅ Docker image built successfully!")
    print("\n📋 Run your app with:")
    print("   docker run -p 8501:8501 lunarv")
    print("\n📋 Or use docker-compose:")
    print("   docker-compose up --build")
    
    return True

def main():
    """Main deployment menu"""
    print("🌙 Welcome to Lunarv Deployment Helper!")
    print("=" * 50)
    
    while True:
        print("\nChoose deployment option:")
        print("1. 🚀 Streamlit Cloud (Recommended)")
        print("2. ☁️ Heroku")
        print("3. 🐳 Docker")
        print("4. 📋 Check Requirements")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            deploy_streamlit_cloud()
        elif choice == '2':
            deploy_heroku()
        elif choice == '3':
            deploy_docker()
        elif choice == '4':
            check_requirements()
        elif choice == '5':
            print("👋 Goodbye! Happy deploying!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
