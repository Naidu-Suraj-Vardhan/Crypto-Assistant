# Multilingual Cryptocurrency Assistant

An AI-powered assistant that can handle cryptocurrency price queries in multiple languages. The system automatically detects non-English queries, translates them, fetches the requested cryptocurrency prices, and responds in English.

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/LLaMA-3.1-orange.svg" alt="LLaMA 3.1">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</div>

## Features

- **Multilingual Support**: Handles queries in any language through automatic language detection and translation
- **Real-time Cryptocurrency Data**: Fetches up-to-date cryptocurrency prices from CoinGecko API
- **Smart Symbol Recognition**: Automatically maps common cryptocurrency symbols (like BTC, ETH, SOL) to their full names
- **User-Friendly Interface**: Clean Gradio web interface accessible from any browser
- **Conversational AI**: Powered by LLaMA 3.1 for natural, contextual responses
- **Performance Optimizations**:
  - **Rate Limiting**: Prevents API abuse with built-in rate limiting
  - **Response Caching**: Implements efficient caching to reduce redundant API calls
  - **Docker Support**: Easy deployment with containerization

## Quick Start

### Using Docker

The simplest way to run the application is using Docker:

```bash
# Build the Docker image
docker build -t crypto-assistant .

# Run the container
docker run -p 8000:8000 --env-file .env crypto-assistant
```

Then visit `http://localhost:8000` in your browser.

### Manual Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   TOGETHER_API_KEY=your_together_ai_key
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Architecture

The application is built with a modular architecture that separates concerns:

### Core Components

1. **Agent System** (`agent.py`): 
   - Manages the conversational flow
   - Handles function calling and tool execution
   - Processes responses from the LLM

2. **Tools** (`tools.py`):
   - `get_crypto_price`: Fetches cryptocurrency prices
   - `translate_text`: Translates non-English queries to English

3. **Main Application** (`main.py`):
   - Configures the agent with system instructions
   - Sets up the Gradio web interface
   - Manages the user interaction flow

### Technologies Used

- **Together AI**: API access to LLaMA 3.1 8B-Instruct-Turbo model
- **CoinGecko API**: Real-time cryptocurrency data
- **Gradio**: Web interface framework
- **Langdetect**: Language detection library
- **Docker**: Containerization for deployment

## Example Usage

The assistant can handle a wide range of cryptocurrency queries in any language:

- "What's the current price of Bitcoin and Ethereum?"
- "¿Cuál es el precio de Solana?"
- "Quel est le prix du Dogecoin maintenant?"
- "Wie viel kostet Cardano?"
- "Bitcoin का वर्तमान मूल्य क्या है?"
- "イーサリアムの価格はいくらですか？"

## Technical Implementation

### Cryptocurrency Price Fetcher

The system uses a sophisticated cryptocurrency price fetcher with:

- Symbol normalization (eth → ethereum, btc → bitcoin)
- Automatic rate limiting (10 requests/minute)
- 5-minute price caching to reduce API load
- Error handling and fallback mechanisms

### Language Processing

For non-English queries, the system:

1. Detects the input language using langdetect
2. Uses LLaMA 3.1 for high-quality translation to English
3. Processes the translated query for cryptocurrency information
4. Returns clear responses with price data

### Response Handling

The system ensures user-friendly responses by:

- Filtering out raw function calls and JSON
- Prioritizing cryptocurrency price information
- Providing clear fallback messages when needed

## Development and Deployment

### Environment Variables

Required environment variables:
- `TOGETHER_API_KEY`: Your Together AI API key

Optional configuration:
- `CACHE_TTL`: Cache time-to-live in seconds (default: 300)
- `RATE_LIMIT_DELAY`: Delay between API calls in seconds (default: 10)

### Docker Configuration

The included Dockerfile:
- Uses Python 3.10 slim as the base image
- Runs as a non-root user for security
- Exposes port 7860 for the Gradio interface
- Sets appropriate environment variables

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add support for more cryptocurrencies
- Enhance the translation capabilities
- Improve the UI/UX of the Gradio interface
- Optimize the performance further

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

If you encounter any issues:

1. Check your API keys in the `.env` file
2. Ensure all dependencies are installed
3. Check the console logs for detailed error messages

## Limitations

- The free LibreTranslate API has rate limits and may occasionally fail for very long texts
- CoinGecko API also has rate limits in the free tier
- Some cryptocurrency symbols might not be recognized by the CoinGecko API
- Translation quality may vary depending on the language and complexity of the text

## Future Improvements

- Add support for more cryptocurrency data (market cap, volume, etc.)
- Implement more robust error handling for API failures
- Add memory persistence between sessions
- Support for cryptocurrency price charts and visualizations
