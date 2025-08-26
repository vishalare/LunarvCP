# 🚀 Lunarv Cloud Deployment Guide

## 🌟 Quick Start - Streamlit Cloud (Recommended)

### Step 1: Prepare Your Code
1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Lunarv Churn Prediction Tool"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/lunarv.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository:**
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (configuration)
   - `churn_model.py`, `data_utils.py`, `visualization_utils.py` (modules)

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/lunarv`
5. Set main file path: `app.py`
6. Click "Deploy!"

**✅ Done! Your app will be live in 2-3 minutes.**

---

## 🐳 Docker Deployment

### Local Testing
```bash
# Build and run locally
docker-compose up --build

# Access at: http://localhost:8501
```

### Deploy to Any Cloud
```bash
# Build image
docker build -t lunarv .

# Run container
docker run -p 8501:8501 lunarv
```

---

## ☁️ Heroku Deployment

### Step 1: Install Heroku CLI
```bash
# Windows
winget install --id=Heroku.HerokuCLI

# macOS
brew tap heroku/brew && brew install heroku

# Linux
curl https://cli-assets.heroku.com/install.sh | sh
```

### Step 2: Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create lunarv-churn-prediction

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

---

## 🚀 Google Cloud Run

### Step 1: Setup Google Cloud
```bash
# Install gcloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Deploy
```bash
# Build and deploy
gcloud run deploy lunarv \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501

# Get your URL
gcloud run services describe lunarv --region us-central1
```

---

## ☁️ AWS Lambda + API Gateway

### Step 1: Install AWS CLI
```bash
# Download from: https://aws.amazon.com/cli/
aws configure
```

### Step 2: Deploy with Serverless Framework
```bash
# Install serverless
npm install -g serverless

# Deploy
serverless deploy
```

---

## 🔧 Environment Variables

Create a `.env` file for local development:
```env
# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/db

# API Keys (if needed)
OPENAI_API_KEY=your_key_here

# Security
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
```

---

## 📊 Monitoring & Scaling

### Streamlit Cloud
- **Auto-scaling**: Built-in
- **Monitoring**: Basic metrics in dashboard
- **Custom Domain**: Available on paid plans

### Heroku
- **Scaling**: `heroku ps:scale web=2`
- **Monitoring**: `heroku logs --tail`
- **Custom Domain**: `heroku domains:add yourdomain.com`

### Google Cloud Run
- **Auto-scaling**: 0 to 1000 instances
- **Monitoring**: Cloud Monitoring
- **Custom Domain**: Available

---

## 💰 Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| **Streamlit Cloud** | ✅ $0/month | $10/month | Quick deployment |
| **Heroku** | ❌ None | $7/month | Full control |
| **Google Cloud Run** | ✅ $0/month | Pay-per-use | Scalability |
| **AWS Lambda** | ✅ $0/month | Pay-per-use | Serverless |

---

## 🚨 Troubleshooting

### Common Issues

1. **Port Binding Error**
   ```bash
   # Fix: Use environment variable
   export STREAMLIT_SERVER_PORT=$PORT
   ```

2. **Memory Issues**
   ```bash
   # Add to requirements.txt
   psutil>=5.8.0
   ```

3. **File Upload Issues**
   ```python
   # In app.py, add:
   st.set_page_config(
       page_title="Lunarv",
       page_icon="🌙",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   ```

### Performance Tips

1. **Optimize Data Loading**
   ```python
   # Use caching
   @st.cache_data
   def load_data(file):
       return pd.read_csv(file)
   ```

2. **Reduce Memory Usage**
   ```python
   # Process data in chunks
   chunk_size = 10000
   for chunk in pd.read_csv(file, chunksize=chunk_size):
       process_chunk(chunk)
   ```

---

## 🎯 Next Steps

1. **Choose your platform** (Streamlit Cloud recommended)
2. **Deploy following the guide above**
3. **Test your deployment**
4. **Set up monitoring**
5. **Configure custom domain** (optional)

---

## 📞 Support

- **Streamlit Cloud**: [docs.streamlit.io](https://docs.streamlit.io)
- **Heroku**: [devcenter.heroku.com](https://devcenter.heroku.com)
- **Google Cloud**: [cloud.google.com/run](https://cloud.google.com/run)
- **AWS**: [aws.amazon.com/lambda](https://aws.amazon.com/lambda)

---

**🎉 Congratulations! Your Lunarv Churn Prediction Tool is now ready for the cloud!**
