import time
import requests
import json
from langdetect import detect
import os
from together import Together

# Initialize Together API client
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
TRANSLATION_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

class CryptoPriceFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.crypto_mapping = {
            # Major abbreviations to their full names in CoinGecko format
            "btc": "bitcoin",
            "eth": "ethereum", 
            "sol": "solana",
            "doge": "dogecoin",
            "xrp": "ripple",
            "bnb": "binancecoin",
            "ada": "cardano",
            "dot": "polkadot",
            "link": "chainlink",
            "ltc": "litecoin",
            "avax": "avalanche-2",
            "matic": "polygon",
            "shib": "shiba-inu",
            "uni": "uniswap",
            "xlm": "stellar",
            
            # Keep full names as-is
            "bitcoin": "bitcoin",
            "ethereum": "ethereum",
            "solana": "solana",
            "dogecoin": "dogecoin",
            "ripple": "ripple",
            "binancecoin": "binancecoin",
            "cardano": "cardano",
            "polkadot": "polkadot",
            "chainlink": "chainlink",
            "litecoin": "litecoin",
            "stellar": "stellar",
            "uniswap": "uniswap"
        }
        # Simple cache with 5-minute expiry
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        self.rate_limit_delay = 10  # seconds
        self.last_call_time = 0

    def normalize_crypto_name(self, crypto):
        """Normalize cryptocurrency name to match CoinGecko's API requirements"""
        crypto_lower = crypto.lower().strip()
        
        # Check if it's in our mapping
        if crypto_lower in self.crypto_mapping:
            return self.crypto_mapping[crypto_lower]
            
        # Return as-is if not in mapping
        return crypto_lower
        
    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()

    def get_price(self, crypto):
        """Get the current price of a cryptocurrency in USD"""
        self._rate_limit()
        # Normalize the crypto name
        normalized_crypto = self.normalize_crypto_name(crypto)
        
        # Check cache first
        current_time = time.time()
        if normalized_crypto in self.cache:
            cache_time, price = self.cache[normalized_crypto]
            if current_time - cache_time < self.cache_expiry:
                print(f"Using cached price for {normalized_crypto}")
                return price
                
        # If not in cache or expired, fetch from API
        try:
            url = f"{self.base_url}/simple/price?ids={normalized_crypto}&vs_currencies=usd"
            print(f"Fetching price from: {url}")
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if normalized_crypto in data and "usd" in data[normalized_crypto]:
                price = data[normalized_crypto]["usd"]
                # Store in cache
                self.cache[normalized_crypto] = (current_time, price)
                return price
            else:
                print(f"Could not find price data for {normalized_crypto}")
                return None
        except Exception as e:
            print(f"Error fetching price for {normalized_crypto}: {e}")
            return None

# Crypto price tool implementation
def get_crypto_price(crypto: str) -> str:
    """
    Get the current price of a cryptocurrency and format the response.
    
    Args:
        crypto (str): The cryptocurrency to check (e.g., 'bitcoin', 'ethereum', 'eth', 'btc')
        
    Returns:
        str: JSON string with price information and formatted message
    """
    print(f"get_crypto_price called with: {crypto}")
    fetcher = CryptoPriceFetcher()
    
    # Get the normalized name for display
    normalized = fetcher.normalize_crypto_name(crypto)
    
    # Get the actual price
    price = fetcher.get_price(normalized)
    
    if price:
        # Use the normalized name in the response for clarity
        display_name = normalized.capitalize()
        return json.dumps({
            "crypto": display_name,
            "price_usd": price,
            "message": f"The current price of {display_name} is ${price}.",
            "status": "success"
        })
    else:
        return json.dumps({
            "crypto": crypto,
            "message": f"Sorry, I couldn't fetch the price for {crypto}. Please check the cryptocurrency name and try again.",
            "status": "error"
        })

class LanguageTranslator:
    def __init__(self):
        self.cache = {}
        self.last_call_time = 0
        self.rate_limit_delay = 5  # seconds

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_call_time = time.time()

    def translate_to_english(self, text):
        if not text:
            return "", "en"
            
        # Check cache
        if text in self.cache:
            return self.cache[text]
            
        try:
            # Detect language
            detected_lang = detect(text)
            
            # If already English, return as is
            if detected_lang == 'en':
                self.cache[text] = (text, detected_lang)
                return text, detected_lang
                
            # Apply rate limiting
            self._rate_limit()
            
            # Use LLaMA 3.1 for translation instead of an external API
            prompt = f"""Translate the following text from {detected_lang} to English. 
            Return ONLY the translated text, nothing else.
            
            Text to translate: "{text}"
            
            Translation:"""
            
            response = together_client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Translate text to English accurately and fluently."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.3,
                presence_penalty=0.5
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Clean up the response - remove quotes if present
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
                
            self.cache[text] = (translated_text, detected_lang)
            return translated_text, detected_lang
            
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original text if translation fails
            return text, "unknown"

# Global translator instance
translator = LanguageTranslator()

def translate_text(text: str) -> str:
    """
    Detect language and translate non-English text to English
    Args:
        text (str): Text to translate
    Returns:
        str: Translated text (or original if already English) and detected language
    """
    translated, detected_lang = translator.translate_to_english(text)
    return json.dumps({
        "translated_text": translated,
        "detected_language": detected_lang,
        "is_english": detected_lang == "en"
    })