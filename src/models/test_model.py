"""
Test script for the cryptocurrency advisor model.
"""

import torch
from transformers import AutoTokenizer
import logging
from typing import Dict, List
import json
from datetime import datetime

from src.models.transformer.crypto_advisor import CryptoAdvisorModel
from src.data.collectors.market_data import MarketDataCollector
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model(
    model_path: str = settings.MODEL_PATH,
    test_queries: List[str] = None
):
    """
    Test the cryptocurrency advisor model with sample queries.
    
    Args:
        model_path: Path to the trained model
        test_queries: List of test queries
    """
    try:
        # Initialize components
        logger.info("Initializing model and market data collector...")
        model = CryptoAdvisorModel.load_model(model_path)
        market_collector = MarketDataCollector()
        
        # Default test queries if none provided
        if test_queries is None:
            test_queries = [
                "What's the current trend for BTC?",
                "Should I invest in ETH now?",
                "What's the market sentiment for BNB?",
                "Is this a good time to buy BTC?",
                "What are the risks of investing in ETH?"
            ]
        
        # Test each query
        for query in test_queries:
            logger.info(f"\nTesting query: {query}")
            
            # Get market data
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            market_data = market_collector.get_market_metrics(symbols)
            
            # Generate advice
            advice, confidence = model.get_advice(
                market_data=market_data,
                user_query=query
            )
            
            # Print results
            print("\n" + "="*80)
            print(f"Query: {query}")
            print("-"*80)
            print(f"Advice: {advice}")
            print("-"*80)
            print("Confidence Metrics:")
            for metric, value in confidence.items():
                print(f"  {metric}: {value:.2f}")
            print("="*80 + "\n")
            
            # Log results
            logger.info(f"Generated advice for query: {query}")
            logger.info(f"Confidence metrics: {confidence}")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

def save_test_results(
    results: List[Dict],
    output_file: str = "test_results.json"
):
    """
    Save test results to a JSON file.
    
    Args:
        results: List of test results
        output_file: Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        logger.info(f"Test results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving test results: {str(e)}")

if __name__ == "__main__":
    # Run tests
    test_model() 