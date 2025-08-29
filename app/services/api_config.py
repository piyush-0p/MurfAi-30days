from fastapi import WebSocket
import json

class ApiConfigManager:
    """Manages API configuration updates during runtime."""
    
    # Class variables to track updates
    updates_count = 0
    
    @staticmethod
    async def update_api_keys(websocket: WebSocket, data: dict):
        """Update API keys from WebSocket message."""
        import main
        import google.generativeai as genai
        import assemblyai as aai
        
        try:
            # Track variables that were successfully updated
            updated_keys = []
            
            # Update AssemblyAI API key
            if data.get('assemblyai') and len(data['assemblyai']) > 10:
                main.ASSEMBLYAI_API_KEY = data['assemblyai']
                aai.settings.api_key = main.ASSEMBLYAI_API_KEY
                updated_keys.append("AssemblyAI")
            
            # Update Gemini API key
            if data.get('gemini') and len(data['gemini']) > 10:
                main.GEMINI_API_KEY = data['gemini']
                genai.configure(api_key=main.GEMINI_API_KEY)
                updated_keys.append("Gemini")
            
            # Update Weather API key
            if data.get('weather') and len(data['weather']) > 10:
                main.WEATHER_API_KEY = data['weather']
                updated_keys.append("Weather")
            
            # Update Tavily API key
            if data.get('tavily') and len(data['tavily']) > 10:
                main.TAVILY_API_KEY = data['tavily']
                updated_keys.append("Tavily")
            
            # Update MurfAI API key
            if data.get('murf') and len(data['murf']) > 10:
                main.MURF_API_KEY = data['murf']
                updated_keys.append("MurfAI")
            
            # Prepare success message
            success_message = f"Updated API keys: {', '.join(updated_keys)}" if updated_keys else "No API keys were updated"
            
            # Send confirmation message back to client
            await websocket.send_text(json.dumps({
                "type": "API_CONFIG_RESPONSE",
                "success": True,
                "message": success_message
            }))
            
            # Increment update counter
            ApiConfigManager.updates_count += 1
            
            print(f"✅ API keys updated ({ApiConfigManager.updates_count}): {', '.join(updated_keys)}")
            return True
            
        except Exception as e:
            # Send error message back to client
            await websocket.send_text(json.dumps({
                "type": "API_CONFIG_RESPONSE",
                "success": False,
                "message": f"Error updating API keys: {str(e)}"
            }))
            
            print(f"❌ Error updating API keys: {str(e)}")
            return False
