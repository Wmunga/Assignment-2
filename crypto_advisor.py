"""
CryptoBuddy - A friendly cryptocurrency advisor chatbot that provides
investment advice based on profitability and sustainability metrics.
"""

from crypto_db import (
    get_all_cryptos,
    get_crypto_info,
    get_trending_cryptos,
    get_sustainable_cryptos,
    get_profitable_cryptos
)

class CryptoBuddy:
    def __init__(self):
        self.name = "CryptoBuddy"
        self.greeting = "👋 Hi! I'm CryptoBuddy, your friendly crypto advisor! How can I help you today?"
        self.help_message = """
I can help you with:
- Trending cryptocurrencies
- Sustainable crypto options
- Profitable investment opportunities
- Information about specific cryptocurrencies
- General crypto advice

Just ask me anything about crypto! 😊
"""
    
    def get_response(self, user_input):
        """Process user input and return appropriate response."""
        user_input = user_input.lower().strip()
        
        # Greeting patterns
        if any(word in user_input for word in ["hi", "hello", "hey", "greetings"]):
            return self.greeting
        
        # Help request
        if "help" in user_input:
            return self.help_message
        
        # Trending cryptocurrencies
        if any(phrase in user_input for phrase in ["trending", "trend", "going up", "rising"]):
            trending = get_trending_cryptos()
            if trending:
                return f"📈 Currently trending cryptocurrencies are: {', '.join(trending)}"
            return "I don't see any cryptocurrencies trending up right now. 😕"
        
        # Sustainable options
        if any(phrase in user_input for phrase in ["sustainable", "green", "environment", "eco"]):
            sustainable = get_sustainable_cryptos()
            if sustainable:
                return f"🌱 The most sustainable cryptocurrencies are: {', '.join(sustainable)}"
            return "I don't have any highly sustainable cryptocurrencies in my database right now. 😕"
        
        # Profitable investments
        if any(phrase in user_input for phrase in ["profitable", "invest", "best", "recommend"]):
            profitable = get_profitable_cryptos()
            if profitable:
                return f"💰 Based on market cap and price trends, these cryptocurrencies look promising: {', '.join(profitable)}"
            return "I don't see any highly profitable cryptocurrencies in my database right now. 😕"
        
        # Specific cryptocurrency information
        for crypto in get_all_cryptos():
            if crypto.lower() in user_input:
                info = get_crypto_info(crypto)
                return f"""
Here's what I know about {crypto} ({info['symbol']}):
📊 Price Trend: {info['price_trend']}
💎 Market Cap: {info['market_cap']}
⚡ Energy Use: {info['energy_use']}
🌍 Sustainability Score: {info['sustainability_score']}/10
📝 {info['description']}
"""
        
        # Default response for unrecognized queries
        return "I'm not sure I understand. Try asking me about trending cryptocurrencies, sustainable options, or specific coins like Bitcoin! 😊"

def main():
    """Main function to run the chatbot."""
    bot = CryptoBuddy()
    print(bot.greeting)
    print(bot.help_message)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print(f"\n{bot.name}: Goodbye! Happy trading! 👋")
                break
            
            response = bot.get_response(user_input)
            print(f"\n{bot.name}: {response}")
            
        except KeyboardInterrupt:
            print(f"\n\n{bot.name}: Goodbye! Happy trading! 👋")
            break
        except Exception as e:
            print(f"\n{bot.name}: Oops! Something went wrong. Please try again! 😅")

if __name__ == "__main__":
    main() 