# ✅ Render Deployment Checklist

## Before You Start
- [ ] Your code is pushed to GitHub
- [ ] You have a Render.com account
- [ ] You have all API keys ready

## API Keys You Need
- [ ] **AssemblyAI API Key** - Get from: https://www.assemblyai.com/
- [ ] **Google Gemini API Key** - Get from: https://aistudio.google.com/app/apikey  
- [ ] **MurfAI API Key** - Get from: https://murf.ai/
- [ ] **OpenWeatherMap API Key** - Get from: https://openweathermap.org/api
- [ ] **Tavily API Key** - Get from: https://tavily.com/

## Deployment Steps

### 1. Create Web Service
- [ ] Go to https://render.com/dashboard
- [ ] Click "New +" → "Web Service"
- [ ] Connect your GitHub repository

### 2. Configure Service
- [ ] **Name**: `murfai-conversational-agent`
- [ ] **Runtime**: `Python 3`
- [ ] **Build Command**: `pip install -r requirements.txt`
- [ ] **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] **Plan**: Free (for testing)

### 3. Environment Variables
Add these in the Environment Variables section:
```
ASSEMBLYAI_API_KEY = [your actual key]
GEMINI_API_KEY = [your actual key]
MURF_API_KEY = [your actual key]
WEATHER_API_KEY = [your actual key]
TAVILY_API_KEY = [your actual key]
PYTHON_VERSION = 3.9.18
```

### 4. Deploy
- [ ] Click "Create Web Service"
- [ ] Wait for build to complete (2-5 minutes)
- [ ] Check deployment logs for any errors

### 5. Test Your App
- [ ] Visit your Render URL
- [ ] Go to `/static/conversation.html`
- [ ] Test API configuration panel
- [ ] Try voice conversation features
- [ ] Test different personas

## Your App URLs
After deployment, your app will be available at:
- **Main App**: `https://your-service-name.onrender.com/static/conversation.html`
- **Health Check**: `https://your-service-name.onrender.com/health`

## Quick Test Commands
```bash
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Test main page
curl https://your-service-name.onrender.com/
```

## Troubleshooting
If deployment fails:
1. Check build logs in Render dashboard
2. Verify all environment variables are set
3. Check that requirements.txt is correct
4. Make sure your GitHub repository is up to date

## Post-Deployment
- [ ] Configure API keys using the integrated panel
- [ ] Test all features (voice, personas, web search, weather)
- [ ] Share your app with others!

## Free Tier Notes
- App sleeps after 15 minutes of inactivity
- 750 hours per month
- To prevent sleeping, upgrade to paid plan ($7/month)

🎉 **You're ready to deploy to Render!**
