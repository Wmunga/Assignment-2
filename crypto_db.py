"""
Cryptocurrency database containing information about various cryptocurrencies.
Each cryptocurrency entry includes data about its price trend, market cap,
energy usage, and sustainability score.
"""

crypto_db = {
    "Bitcoin": {
        "symbol": "BTC",
        "price_trend": "rising",
        "market_cap": "high",
        "energy_use": "high",
        "sustainability_score": 4.5,  # out of 10
        "description": "The first and most well-known cryptocurrency, known for its high energy consumption but strong market presence."
    },
    "Ethereum": {
        "symbol": "ETH",
        "price_trend": "stable",
        "market_cap": "high",
        "energy_use": "medium",
        "sustainability_score": 6.0,  # out of 10
        "description": "A decentralized platform that enables smart contracts and decentralized applications."
    },
    "Cardano": {
        "symbol": "ADA",
        "price_trend": "rising",
        "market_cap": "medium",
        "energy_use": "low",
        "sustainability_score": 8.5,  # out of 10
        "description": "A proof-of-stake blockchain platform that aims to be more sustainable and scalable."
    }
}

def get_all_cryptos():
    """Returns a list of all cryptocurrencies in the database."""
    return list(crypto_db.keys())

def get_crypto_info(crypto_name):
    """Returns information about a specific cryptocurrency."""
    return crypto_db.get(crypto_name)

def get_trending_cryptos():
    """Returns cryptocurrencies with rising price trends."""
    return [crypto for crypto, data in crypto_db.items() 
            if data["price_trend"] == "rising"]

def get_sustainable_cryptos():
    """Returns cryptocurrencies with high sustainability scores."""
    return [crypto for crypto, data in crypto_db.items() 
            if data["sustainability_score"] >= 7.0]

def get_profitable_cryptos():
    """Returns cryptocurrencies with high market cap and rising trends."""
    return [crypto for crypto, data in crypto_db.items() 
            if data["market_cap"] == "high" and data["price_trend"] == "rising"] 