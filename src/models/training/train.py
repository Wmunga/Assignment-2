"""
Training script for the cryptocurrency advisor model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import os
import json
from datetime import datetime
import mlflow
from tqdm import tqdm

from src.config.settings import settings
from src.data.collectors.market_data import MarketDataCollector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoDataset(Dataset):
    """Dataset for cryptocurrency advisory training."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """
        Initialize the dataset.
        
        Args:
            data: List of training examples
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        example = self.data[idx]
        
        # Format input text
        input_text = self._format_example(example)
        
        # Tokenize
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze()
        }
    
    def _format_example(self, example: Dict) -> str:
        """Format a training example as text."""
        market_data = example["market_data"]
        query = example["query"]
        response = example["response"]
        
        # Format market data
        market_info = " ".join([
            f"<{k}>{v:.2f}</{k}>" for k, v in market_data.items()
        ])
        
        # Combine into prompt
        prompt = f"""
Market Data:
{market_info}

User: {query}

Assistant: {response}"""
        
        return prompt

def prepare_training_data(
    market_collector: MarketDataCollector,
    num_examples: int = 1000
) -> List[Dict]:
    """
    Prepare training data from market data and templates.
    
    Args:
        market_collector: Market data collector
        num_examples: Number of examples to generate
        
    Returns:
        List of training examples
    """
    # Load templates
    templates = [
        {
            "query": "What's the current trend for {symbol}?",
            "response_template": "Based on the current market data, {symbol} is showing {trend} trend. The price has {change} by {change_pct}% in the last 24 hours, with a trading volume of {volume}."
        },
        {
            "query": "Should I invest in {symbol} now?",
            "response_template": "Given the current market conditions, {symbol} {advice}. The price is {price_desc}, and the market shows {volatility} volatility. {risk_warning}"
        },
        {
            "query": "What's the market sentiment for {symbol}?",
            "response_template": "The market sentiment for {symbol} appears to be {sentiment}. This is based on the {price_action} price action and {volume_desc} trading volume."
        }
    ]
    
    # Generate examples
    examples = []
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    for _ in tqdm(range(num_examples), desc="Generating training examples"):
        # Get market data
        symbol = np.random.choice(symbols)
        market_data = market_collector.get_market_metrics([symbol])[symbol]
        
        # Select template
        template = np.random.choice(templates)
        
        # Generate response
        trend = "bullish" if market_data["change_24h"] > 0 else "bearish"
        change = "increased" if market_data["change_24h"] > 0 else "decreased"
        change_pct = abs(market_data["change_24h"])
        volume = f"${market_data['volume']:,.2f}"
        
        price_desc = "relatively high" if market_data["price"] > market_data["high_24h"] * 0.95 else "at a good entry point"
        volatility = "high" if market_data.get("volatility", 0) > 0.5 else "moderate"
        risk_warning = "Please note that cryptocurrency investments carry significant risk." if volatility == "high" else "As always, invest only what you can afford to lose."
        
        sentiment = "positive" if market_data["change_24h"] > 2 else "neutral" if market_data["change_24h"] > -2 else "negative"
        price_action = "strong" if abs(market_data["change_24h"]) > 5 else "moderate"
        volume_desc = "high" if market_data["volume"] > market_data.get("avg_volume", 0) * 1.5 else "normal"
        
        advice = "might be a good time to consider an investment" if trend == "bullish" else "it might be better to wait for a better entry point"
        
        # Format response
        response = template["response_template"].format(
            symbol=symbol.split('/')[0],
            trend=trend,
            change=change,
            change_pct=change_pct,
            volume=volume,
            price_desc=price_desc,
            volatility=volatility,
            risk_warning=risk_warning,
            sentiment=sentiment,
            price_action=price_action,
            volume_desc=volume_desc,
            advice=advice
        )
        
        # Create example
        example = {
            "market_data": market_data,
            "query": template["query"].format(symbol=symbol.split('/')[0]),
            "response": response
        }
        
        examples.append(example)
    
    return examples

def train_model(
    model_name: str = settings.MODEL_NAME,
    output_dir: str = settings.MODEL_PATH,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 100
):
    """
    Train the cryptocurrency advisor model.
    
    Args:
        model_name: Base model to use
        output_dir: Directory to save the model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps for gradient accumulation
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Number of steps between logging
        save_steps: Number of steps between model saves
    """
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("crypto_advisor_training")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_name": model_name,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps
            })
            
            # Load tokenizer and model
            logger.info(f"Loading model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add special tokens
            special_tokens = {
                "additional_special_tokens": [
                    "<BTC>", "<ETH>", "<USDT>", "<BNB>",
                    "<PRICE>", "<VOLUME>", "<MARKET_CAP>",
                    "<ADVICE>", "<RISK>", "<TREND>"
                ]
            }
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            # Prepare training data
            logger.info("Preparing training data...")
            market_collector = MarketDataCollector()
            train_data = prepare_training_data(market_collector)
            
            # Create dataset
            dataset = CryptoDataset(train_data, tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                save_steps=save_steps,
                save_total_limit=2,
                evaluation_strategy="no",
                load_best_model_at_end=False,
                report_to="none"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )
            )
            
            # Train model
            logger.info("Starting training...")
            trainer.train()
            
            # Save model
            logger.info(f"Saving model to {output_dir}...")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Log model
            mlflow.pytorch.log_model(model, "model")
            
            logger.info("Training completed successfully")
            
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 