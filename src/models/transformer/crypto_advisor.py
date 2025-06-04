"""
CryptoAdvisor: A transformer-based model for cryptocurrency advisory.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class CryptoAdvisorModel(nn.Module):
    """Main model class for the cryptocurrency advisor."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the CryptoAdvisor model.
        
        Args:
            model_name: Base model to use (default: gpt2)
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Load base model and tokenizer
        logger.info(f"Loading model {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Add special tokens for cryptocurrency domain
        special_tokens = {
            "additional_special_tokens": [
                "<BTC>", "<ETH>", "<USDT>", "<BNB>",
                "<PRICE>", "<VOLUME>", "<MARKET_CAP>",
                "<ADVICE>", "<RISK>", "<TREND>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model initialized successfully")
    
    def prepare_input(
        self,
        market_data: Dict[str, float],
        user_query: str,
        context: Optional[List[str]] = None
    ) -> str:
        """
        Prepare input prompt from market data and user query.
        
        Args:
            market_data: Dictionary of market metrics
            user_query: User's question or request
            context: Optional conversation history
            
        Returns:
            Formatted input prompt
        """
        # Format market data
        market_info = " ".join([
            f"<{k}>{v:.2f}</{k}>" for k, v in market_data.items()
        ])
        
        # Format context if provided
        context_str = ""
        if context:
            context_str = "\n".join([
                f"Previous: {msg}" for msg in context[-3:]  # Last 3 messages
            ]) + "\n"
        
        # Combine into final prompt
        prompt = f"""
Market Data:
{market_info}

{context_str}User: {user_query}

Assistant:"""
        return prompt
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 200
    ) -> str:
        """
        Generate a response using the model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def get_advice(
        self,
        market_data: Dict[str, float],
        user_query: str,
        context: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Get cryptocurrency advice based on market data and user query.
        
        Args:
            market_data: Current market metrics
            user_query: User's question
            context: Optional conversation history
            
        Returns:
            Tuple of (advice_text, confidence_metrics)
        """
        # Prepare input
        prompt = self.prepare_input(market_data, user_query, context)
        
        # Generate response
        response = self.generate_response(prompt)
        
        # Extract confidence metrics (placeholder for now)
        confidence_metrics = {
            "market_confidence": 0.85,
            "trend_confidence": 0.75,
            "risk_level": 0.3
        }
        
        return response, confidence_metrics
    
    def save_model(self, path: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, **kwargs):
        """Load model from disk."""
        model = cls(**kwargs)
        model.model = AutoModelForCausalLM.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model.to(model.device)
        logger.info(f"Model loaded from {path}")
        return model