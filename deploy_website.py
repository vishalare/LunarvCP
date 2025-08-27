#!/usr/bin/env python3
"""
🌐 Website Deployment Script for Lunarv
Deploy your churn prediction tool on your own website/server
"""

import os
import subprocess
import sys
from pathlib import Path

def create_nginx_config():
    """Create Nginx configuration for Lunarv"""
    config = """server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}"""
    
    with open('nginx_lunarv.conf', 'w') as f:
        f.write(config)
    print("✅ Created nginx_lunarv.conf")
    return config

def create_systemd_service():
    """Create systemd service file for Lunarv"""
    service = """[Unit]
Description=Lunarv Churn Prediction App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/lunarv
Environment=PATH=/var/www/lunarv/venv/bin
ExecStart=/var/www/lunarv/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"""
    
    with open('lunarv.service', 'w') as f:
        f.write(service)
    print("✅ Created lunarv.service")
    return service

def create_docker_website():
    """Create Docker setup for website deployment"""
    dockerfile = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    nginx \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy nginx config
COPY nginx_lunarv.conf /etc/nginx/sites-available/lunarv
RUN ln -s /etc/nginx/sites-available/lunarv /etc/nginx/sites-enabled/
RUN rm /etc/nginx/sites-enabled/default

# Expose ports
EXPOSE 80 8501

# Start nginx and streamlit
COPY start_services.sh /start_services.sh
RUN chmod +x /start_services.sh
CMD ["/start_services.sh"]"""
    
    with open('Dockerfile.website', 'w') as f:
        f.write(dockerfile)
    print("✅ Created Dockerfile.website")
    
    # Create start script
    start_script = """#!/bin/bash
# Start nginx
service nginx start

# Start Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# Keep container running
tail -f /dev/null"""
    
    with open('start_services.sh', 'w') as f:
        f.write(start_script)
    print("✅ Created start_services.sh")

def create_docker_compose_website():
    """Create Docker Compose for website deployment"""
    compose = """version: '3.8'

services:
  lunarv:
    build:
      context: .
      dockerfile: Dockerfile.website
    ports:
      - "80:80"
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0

  # Optional: Add a database for storing results
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: lunarv
      POSTGRES_USER: lunarv_user
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:"""
    
    with open('docker-compose.website.yml', 'w') as f:
        f.write(compose)
    print("✅ Created docker-compose.website.yml")

def create_apache_config():
    """Create Apache configuration for Lunarv"""
    config = """<VirtualHost *:80>
    ServerName your-domain.com
    DocumentRoot /var/www/html
    
    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8501/
    ProxyPassReverse / http://127.0.0.1:8501/
    
    ErrorLog ${APACHE_LOG_DIR}/lunarv_error.log
    CustomLog ${APACHE_LOG_DIR}/lunarv_access.log combined
</VirtualHost>"""
    
    with open('apache_lunarv.conf', 'w') as f:
        f.write(config)
    print("✅ Created apache_lunarv.conf")
    return config

def create_website_integration():
    """Create HTML integration files"""
    # Simple HTML page
    html_page = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lunarv - AI Churn Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .app-frame { border: none; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
        .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌙 Lunarv - AI Churn Prediction Platform</h1>
        <p>Predict customer churn with 95%+ accuracy using advanced machine learning</p>
    </div>
    
    <div class="container">
        <div class="info">
            <h3>🚀 How to Use:</h3>
            <ol>
                <li><strong>Upload Data:</strong> Upload your customer dataset (CSV, Excel)</li>
                <li><strong>Configure Columns:</strong> Map your customer ID and churn columns</li>
                <li><strong>Train Model:</strong> Let AI analyze your data and build predictions</li>
                <li><strong>Get Results:</strong> View churn probabilities and insights</li>
                <li><strong>Export:</strong> Download reports and predictions</li>
            </ol>
        </div>
        
        <iframe 
            src="http://localhost:8501" 
            width="100%" 
            height="800px" 
            class="app-frame"
            title="Lunarv Churn Prediction App">
        </iframe>
    </div>
</body>
</html>"""
    
    with open('index.html', 'w') as f:
        f.write(html_page)
    print("✅ Created index.html")
    
    # JavaScript integration
    js_integration = """// Lunarv Website Integration JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Auto-resize iframe based on content
    const iframe = document.querySelector('iframe');
    
    if (iframe) {
        iframe.addEventListener('load', function() {
            // You can add custom integration logic here
            console.log('Lunarv app loaded successfully');
        });
    }
    
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.innerHTML = '<p>Loading Lunarv...</p>';
    loadingDiv.style.textAlign = 'center';
    loadingDiv.style.padding = '20px';
    
    iframe.parentNode.insertBefore(loadingDiv, iframe);
    
    iframe.addEventListener('load', function() {
        loadingDiv.style.display = 'none';
    });
});"""
    
    with open('lunarv-integration.js', 'w') as f:
        f.write(js_integration)
    print("✅ Created lunarv-integration.js")

def main():
    """Main deployment setup"""
    print("🌐 Lunarv Website Deployment Setup")
    print("=" * 50)
    
    # Create all deployment files
    create_nginx_config()
    create_systemd_service()
    create_docker_website()
    create_docker_compose_website()
    create_apache_config()
    create_website_integration()
    
    print("\n🎉 Website deployment files created successfully!")
    print("\n📁 Files created:")
    print("  • nginx_lunarv.conf - Nginx configuration")
    print("  • apache_lunarv.conf - Apache configuration")
    print("  • lunarv.service - Systemd service")
    print("  • Dockerfile.website - Docker setup")
    print("  • docker-compose.website.yml - Docker Compose")
    print("  • index.html - Sample HTML page")
    print("  • lunarv-integration.js - JavaScript integration")
    
    print("\n🚀 Deployment Options:")
    print("1. Docker (Recommended): docker-compose -f docker-compose.website.yml up -d")
    print("2. Nginx: Copy nginx_lunarv.conf to /etc/nginx/sites-available/")
    print("3. Apache: Copy apache_lunarv.conf to /etc/apache2/sites-available/")
    print("4. Systemd: Copy lunarv.service to /etc/systemd/system/")
    
    print("\n💡 Quick Start (Docker):")
    print("  docker-compose -f docker-compose.website.yml up -d")
    print("  # Then visit http://localhost")

if __name__ == "__main__":
    main()
