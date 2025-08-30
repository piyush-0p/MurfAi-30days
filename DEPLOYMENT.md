# Vercel Deployment Guide

## Prerequisites
1. A Vercel account (sign up at https://vercel.com)
2. Vercel CLI installed: `npm install -g vercel`
3. Your API keys ready

## API Keys Required
- **AssemblyAI**: For speech recognition - Get from https://www.assemblyai.com/
- **Google Gemini**: For AI language model - Get from https://aistudio.google.com/app/apikey
- **MurfAI**: For text-to-speech - Get from https://murf.ai/
- **OpenWeatherMap**: For weather data - Get from https://openweathermap.org/api
- **Tavily**: For web search - Get from https://tavily.com/

## Deployment Steps

### Option 1: Deploy via Vercel CLI

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy the project**:
   ```bash
   vercel
   ```
   - Follow the prompts
   - Choose "yes" when asked to link to existing project or create new one
   - Select your scope (personal account or team)

4. **Set Environment Variables**:
   ```bash
   vercel env add ASSEMBLYAI_API_KEY
   vercel env add GEMINI_API_KEY
   vercel env add MURF_API_KEY
   vercel env add WEATHER_API_KEY
   vercel env add TAVILY_API_KEY
   ```

### Option 2: Deploy via Vercel Dashboard

1. **Connect GitHub Repository**:
   - Go to https://vercel.com/dashboard
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Environment Variables**:
   - In the project settings, go to "Environment Variables"
   - Add all required API keys:
     - `ASSEMBLYAI_API_KEY`
     - `GEMINI_API_KEY` 
     - `MURF_API_KEY`
     - `WEATHER_API_KEY`
     - `TAVILY_API_KEY`

3. **Deploy**:
   - Click "Deploy"
   - Wait for the build to complete

## Important Notes

### WebSocket Limitations
- **Vercel has limitations with WebSockets** in serverless functions
- WebSocket connections may not work as expected on Vercel
- Consider using Server-Sent Events (SSE) or polling as alternatives
- For full WebSocket support, consider deploying to:
  - Railway.app
  - Render.com  
  - Digital Ocean App Platform
  - AWS ECS/Lambda with WebSocket API Gateway

### File Upload Limitations
- Vercel has a 4.5MB limit for serverless function payloads
- Large audio file uploads may fail
- Consider using external storage (AWS S3, Cloudinary) for file handling

### Cold Starts
- First request after inactivity may be slower due to cold starts
- Consider implementing health check endpoints

## Alternative Deployment Options

If you encounter issues with Vercel due to WebSocket requirements:

### Railway.app (Recommended for WebSockets)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Render.com
- Create account at https://render.com
- Connect GitHub repository
- Choose "Web Service"
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Troubleshooting

1. **Build Errors**: Check that all dependencies in requirements.txt are compatible
2. **API Errors**: Verify all environment variables are set correctly
3. **WebSocket Issues**: Consider switching to alternative deployment platform
4. **Timeout Errors**: Check function timeout limits in vercel.json

## Post-Deployment

1. **Test the deployment**: Visit your Vercel URL
2. **Configure API keys**: Use the integrated API configuration panel
3. **Test voice features**: Ensure microphone permissions work
4. **Monitor logs**: Use Vercel dashboard to monitor function logs
