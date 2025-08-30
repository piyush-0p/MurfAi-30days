# 🚀 Render.com Deployment Guide for MurfAI Conversational Agent

## Why Render?
- ✅ **Full WebSocket Support** - Perfect for real-time audio streaming
- ✅ **Easy Deployment** - Simple GitHub integration
- ✅ **Free Tier Available** - Great for testing
- ✅ **Persistent Storage** - For file uploads
- ✅ **Custom Domains** - Professional URLs
- ✅ **Automatic SSL** - HTTPS by default

## 📋 Prerequisites

1. **GitHub Account** - Your code should be on GitHub
2. **Render Account** - Sign up at https://render.com (free)
3. **API Keys Ready** - Get your API keys from:
   - **AssemblyAI**: https://www.assemblyai.com/
   - **Google Gemini**: https://aistudio.google.com/app/apikey
   - **MurfAI**: https://murf.ai/
   - **OpenWeatherMap**: https://openweathermap.org/api
   - **Tavily**: https://tavily.com/

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository
First, make sure your latest changes are pushed to GitHub:

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### Step 2: Create New Web Service on Render

1. **Go to Render Dashboard**
   - Visit https://render.com/dashboard
   - Click **"New +"** button
   - Select **"Web Service"**

2. **Connect GitHub Repository**
   - Choose **"Build and deploy from a Git repository"**
   - Click **"Connect"** next to GitHub
   - Find and select your `MurfAi-30days` repository
   - Click **"Connect"**

### Step 3: Configure Web Service Settings

Fill in these settings on Render:

**Basic Info:**
- **Name**: `murfai-conversational-agent` (or your preferred name)
- **Region**: `Oregon (US West)` (or closest to you)
- **Branch**: `main`

**Build & Deploy:**
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Plan:**
- Choose **"Free"** for testing (can upgrade later)

### Step 4: Set Environment Variables

In the **Environment Variables** section, add these variables:

```
ASSEMBLYAI_API_KEY = your_assemblyai_api_key_here
GEMINI_API_KEY = your_gemini_api_key_here  
MURF_API_KEY = your_murf_api_key_here
WEATHER_API_KEY = your_weather_api_key_here
TAVILY_API_KEY = your_tavily_api_key_here
CO_API_KEY = dummy-key-for-tavily
PYTHON_VERSION = 3.9.18
```

**⚠️ Important**: Replace `your_*_api_key_here` with your actual API keys!

### Step 5: Advanced Settings (Optional)

**Auto-Deploy:**
- ✅ Enable **"Auto-Deploy"** - Automatically deploy when you push to GitHub

**Health Check Path:**
- Set to `/` (root path)

**Persistent Disk (Optional):**
- If you want file uploads to persist, add a disk:
  - **Name**: `uploads`
  - **Mount Path**: `/opt/render/project/src/uploads`
  - **Size**: `1 GB`

### Step 6: Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. Monitor the build logs in real-time
4. Wait for deployment to complete (usually 2-5 minutes)

### Step 7: Test Your Deployment

1. **Get Your URL**: Render will provide a URL like `https://your-service-name.onrender.com`
2. **Visit Your App**: Go to `https://your-service-name.onrender.com/static/conversation.html`
3. **Test Features**:
   - Configure your API keys using the integrated panel
   - Test voice conversation features
   - Try different personas
   - Test web search and weather features

## 🔧 Post-Deployment Configuration

### Configure Your API Keys
1. Visit your deployed app
2. Use the **API Configuration** panel we integrated
3. Enter all your API keys
4. Click **"Save Configuration"**

### Custom Domain (Optional)
If you have a custom domain:
1. Go to your service settings
2. Click **"Custom Domains"**
3. Add your domain
4. Update your DNS settings as instructed

## 🐛 Troubleshooting

### Build Failures
```bash
# Check build logs in Render dashboard
# Common issues:
- Missing dependencies in requirements.txt
- Python version conflicts
- API key formatting errors
```

### Runtime Errors
```bash
# Check application logs in Render dashboard
# Common issues:
- Missing environment variables
- Invalid API keys
- File permission issues
```

### WebSocket Connection Issues
```bash
# WebSockets should work fine on Render
# If you see connection issues:
- Check browser console for errors
- Verify HTTPS is being used (not HTTP)
- Check that your domain allows WebSocket connections
```

### Free Tier Limitations
- **Sleep after 15 minutes** of inactivity
- **750 hours/month** of runtime
- **Limited CPU and memory**

To prevent sleep:
- Upgrade to paid plan ($7/month)
- Or use a service like UptimeRobot to ping your app

## 📊 Monitoring Your App

### View Logs
1. Go to your service dashboard
2. Click **"Logs"** tab
3. Monitor real-time application logs

### Metrics
1. Click **"Metrics"** tab
2. Monitor CPU, memory, and response times

### Events
1. Click **"Events"** tab  
2. See deployment history and status changes

## 🔄 Updating Your App

Whenever you make changes:

```bash
git add .
git commit -m "Update: your changes description"
git push origin main
```

Render will automatically deploy your changes (if auto-deploy is enabled).

## 💡 Pro Tips

1. **Environment Variables**: Keep your API keys secure - never commit them to GitHub
2. **Logging**: Use the integrated logging to debug issues
3. **Health Checks**: Your app responds to `/` for health checks
4. **Static Files**: Your static files are served from `/static/`
5. **WebSocket URL**: Use `wss://` (not `ws://`) for secure WebSocket connections in production

## 🚨 Important Notes

- **First Request**: May be slow due to cold start on free tier
- **File Uploads**: Use persistent disk if you need file uploads to survive deployments
- **Database**: Consider adding PostgreSQL if you need persistent data storage
- **Scaling**: Can scale to multiple instances on paid plans

## 🆘 Need Help?

- **Render Docs**: https://render.com/docs
- **Community**: https://community.render.com
- **Support**: Available through Render dashboard

Your AI Conversational Agent is now ready for the world! 🌟
