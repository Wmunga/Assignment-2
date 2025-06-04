"""
Market data collector for cryptocurrency metrics.
"""

import ccxt
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects real-time and historical cryptocurrency market data."""
    
    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize the market data collector.
        
        Args:
            exchange_id: Exchange to use (default: binance)
            api_key: Optional API key for authenticated requests
            api_secret: Optional API secret for authenticated requests
        """
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        logger.info(f"Initialized market data collector for {exchange_id}")
    
    @lru_cache(maxsize=100)
    def get_ticker(self, symbol: str) -> Dict[str, float]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary of ticker data
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'price': float(ticker['last']),
                'volume': float(ticker['baseVolume']),
                'high_24h': float(ticker['high']),
                'low_24h': float(ticker['low']),
                'change_24h': float(ticker['percentage']),
                'bid': float(ticker['bid']),
                'ask': float(ticker['ask'])
            }
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            return {}
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[Union[str, datetime]] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (e.g., '1h', '1d')
            since: Start time (optional)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if isinstance(since, str):
                since = datetime.fromisoformat(since)
            elif since is None:
                since = datetime.now() - timedelta(days=7)
            
            since_timestamp = int(since.timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_timestamp,
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_cap(self, symbol: str) -> float:
        """
        Get market capitalization for a cryptocurrency.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market capitalization in USD
        """
        try:
            ticker = self.get_ticker(symbol)
            if not ticker:
                return 0.0
            
            # Get circulating supply (this is a placeholder - you'll need to
            # implement proper supply tracking or use an API that provides it)
            supply = self._get_circulating_supply(symbol.split('/')[0])
            return ticker['price'] * supply
            
        except Exception as e:
            logger.error(f"Error calculating market cap for {symbol}: {str(e)}")
            return 0.0
    
    def _get_circulating_supply(self, coin: str) -> float:
        """
        Get circulating supply for a cryptocurrency.
        This is a placeholder - implement proper supply tracking.
        
        Args:
            coin: Cryptocurrency symbol
            
        Returns:
            Circulating supply
        """
        # Placeholder values - replace with actual API calls
        supply_map = {
            'BTC': 19_000_000,
            'ETH': 120_000_000,
            'BNB': 160_000_000,
            'USDT': 80_000_000_000
        }
        return supply_map.get(coin, 0.0)
    
    def get_market_metrics(
        self,
        symbols: List[str] = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    ) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive market metrics for multiple symbols.
        
        Args:
            symbols: List of trading pairs to fetch
            
        Returns:
            Dictionary of market metrics for each symbol
        """
        metrics = {}
        for symbol in symbols:
            try:
                ticker = self.get_ticker(symbol)
                if not ticker:
                    continue
                
                market_cap = self.get_market_cap(symbol)
                metrics[symbol] = {
                    **ticker,
                    'market_cap': market_cap,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                logger.error(f"Error getting metrics for {symbol}: {str(e)}")
                continue
        
        return metrics
    
    def get_historical_volatility(
        self,
        symbol: str,
        days: int = 30,
        timeframe: str = '1d'
    ) -> float:
        """
        Calculate historical volatility.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days to consider
            timeframe: Candle timeframe
            
        Returns:
            Historical volatility (annualized)
        """
        try:
            df = self.get_ohlcv(
                symbol,
                timeframe=timeframe,
                since=datetime.now() - timedelta(days=days)
            )
            
            if df.empty:
                return 0.0
            
            # Calculate daily returns
            df['returns'] = df['close'].pct_change()
            
            # Calculate volatility (annualized)
            volatility = df['returns'].std() * (252 ** 0.5)  # Annualized
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.0