"""
Main API server for the cryptocurrency advisor.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime
import json

from src.models.transformer.crypto_advisor import CryptoAdvisorModel
from src.data.collectors.market_data import MarketDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CryptoAdvisor API",
    description="AI-powered cryptocurrency advisory system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/crypto_advisor")
API_KEY = os.getenv("EXCHANGE_API_KEY")
API_SECRET = os.getenv("EXCHANGE_API_SECRET")

# Initialize components
try:
    model = CryptoAdvisorModel.load_model(MODEL_PATH)
    market_data = MarketDataCollector(api_key=API_KEY, api_secret=API_SECRET)
    logger.info("Successfully initialized model and market data collector")
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

# Pydantic models for request/response
class UserQuery(BaseModel):
    query: str = Field(..., description="User's question or request")
    context: Optional[List[str]] = Field(None, description="Previous conversation context")
    symbols: Optional[List[str]] = Field(
        default=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        description="Cryptocurrency symbols to analyze"
    )

class AdviceResponse(BaseModel):
    advice: str = Field(..., description="Generated advice")
    confidence_metrics: Dict[str, float] = Field(..., description="Confidence scores")
    market_data: Dict[str, Dict[str, float]] = Field(..., description="Current market metrics")
    timestamp: datetime = Field(..., description="Response timestamp")

# Store conversation history (in-memory for now)
conversation_history: Dict[str, List[Dict]] = {}

def get_conversation_history(user_id: str) -> List[str]:
    """Get conversation history for a user."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return [msg["query"] for msg in conversation_history[user_id]]

def update_conversation_history(user_id: str, query: str, response: str):
    """Update conversation history for a user."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 10 messages
    conversation_history[user_id] = conversation_history[user_id][-10:]

@app.post("/advice", response_model=AdviceResponse)
async def get_advice(
    query: UserQuery,
    background_tasks: BackgroundTasks,
    user_id: str = "default"
):
    """
    Get cryptocurrency advice based on user query and market data.
    
    Args:
        query: User's question and context
        background_tasks: FastAPI background tasks
        user_id: User identifier (for conversation history)
        
    Returns:
        Advice response with market data and confidence metrics
    """
    try:
        # Get market data
        market_metrics = market_data.get_market_metrics(query.symbols)
        if not market_metrics:
            raise HTTPException(
                status_code=503,
                detail="Unable to fetch market data"
            )
        
        # Get conversation history
        context = get_conversation_history(user_id)
        
        # Generate advice
        advice, confidence = model.get_advice(
            market_data=market_metrics,
            user_query=query.query,
            context=context
        )
        
        # Prepare response
        response = AdviceResponse(
            advice=advice,
            confidence_metrics=confidence,
            market_data=market_metrics,
            timestamp=datetime.now()
        )
        
        # Update conversation history in background
        background_tasks.add_task(
            update_conversation_history,
            user_id,
            query.query,
            advice
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating advice: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating advice: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "market_data_available": market_data is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """
    Get current market data for a specific symbol.
    
    Args:
        symbol: Trading pair symbol
        
    Returns:
        Current market metrics
    """
    try:
        metrics = market_data.get_market_metrics([symbol])
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {symbol}"
            )
        return metrics[symbol]
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching market data: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )