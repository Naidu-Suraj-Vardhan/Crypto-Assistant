import json
import os
import gradio as gr
from typing import List
from dotenv import load_dotenv
from agent import Agent, AgentExecutor
from tools import get_crypto_price, translate_text

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"  # Listen on all interfaces for Docker compatibility

# Get configuration from environment variables
MODEL = os.getenv("MODEL", DEFAULT_MODEL)
PORT = int(os.getenv("PORT", DEFAULT_PORT))
HOST = os.getenv("HOST", DEFAULT_HOST)
SHARE = os.getenv("SHARE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Set up logging based on DEBUG flag
if DEBUG:
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Debug mode enabled")

# Initialize components once
def initialize_agent():
    # Define agent configuration
    instructions = """You are a multilingual cryptocurrency price assistant. Your PRIMARY PURPOSE is to provide cryptocurrency prices when asked, but you also need to be friendly and conversational.

CRITICAL INSTRUCTIONS:
1. ALWAYS USE get_crypto_price FUNCTION FOR ANY PRICE QUERY
2. ALWAYS USE translate_text FUNCTION FOR NON-ENGLISH QUERIES FIRST

CONVERSATION HANDLING:
- Respond naturally to greetings (hello, hi, hey, etc.)
- Engage in basic small talk while being helpful and friendly
- If the user isn't asking about cryptocurrencies, respond conversationally
- Remember that you are helpful and friendly, not just a price lookup tool

CRYPTOCURRENCY PRICE REQUESTS:
1. ANY mention of a cryptocurrency with intent to know its price MUST trigger get_crypto_price
2. NEVER skip calling get_crypto_price for price queries
3. ALWAYS recognize abbreviations (eth, btc, sol, etc.) as cryptocurrencies
4. For multiple cryptocurrencies, call get_crypto_price for EACH ONE
5. NEVER guess or hallucinate prices - ALWAYS use the function

NON-ENGLISH QUERIES:
1. When you receive a message in a non-English language, FIRST use translate_text
2. After translation, process the English version to understand the request
3. If it's a cryptocurrency price query, then call get_crypto_price
4. Respond in English, but acknowledge you understood their original language

Common cryptocurrency identifiers (ALWAYS trigger function calls):
- "eth", "ethereum" → use get_crypto_price(crypto="ethereum")
- "btc", "bitcoin" → use get_crypto_price(crypto="bitcoin") 
- "sol", "solana" → use get_crypto_price(crypto="solana")
- "doge", "dogecoin" → use get_crypto_price(crypto="dogecoin")
- "xrp", "ripple" → use get_crypto_price(crypto="ripple")
- "bnb" → use get_crypto_price(crypto="binancecoin")
- "ada", "cardano" → use get_crypto_price(crypto="cardano")
- "dot", "polkadot" → use get_crypto_price(crypto="polkadot")

Examples of price queries requiring get_crypto_price:
- "What's the price of eth?"
- "How much is bitcoin worth?"
- "Tell me eth price"
- "What's the price of eth or sol?"

Examples of conversation requiring friendly responses:
- "Hello" → "Hi there! I'm your cryptocurrency assistant. How can I help you today?"
- "How are you?" → "I'm doing well, thanks for asking! I'm ready to help with any cryptocurrency price information you need."
- "Thank you" → "You're welcome! Feel free to ask about any cryptocurrency prices anytime."

Function call protocol:
1. For NON-ENGLISH: ALWAYS call translate_text FIRST
2. For PRICE REQUESTS: ALWAYS call get_crypto_price when cryptocurrencies are mentioned with price intent
3. For GREETINGS/CHAT: Respond conversationally without calling functions

Remember: Be helpful and friendly while providing accurate cryptocurrency information.
"""

    agent = Agent(
        name="CryptoExpert",
        model=MODEL,
        instructions=instructions,
        functions=[get_crypto_price, translate_text]
    )
    
    return agent

agent_executor = AgentExecutor()
crypto_agent = initialize_agent()

# For pretty printing complex objects
def safe_print(obj, prefix=""):
    try:
        print(f"{prefix} {json.dumps(obj, indent=2)}")
    except:
        print(f"{prefix} [Could not format object for printing]")

def format_message(message):
    """Convert agent message to display format"""
    try:
        print(f"Formatting message: {message}")
        if message["role"] == "assistant":
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # For debugging only - not shown to user
            if tool_calls:
                calls = "\n".join([
                    f"Called {call['function']['name']}({call['function']['arguments']})"
                    for call in tool_calls if isinstance(call, dict) and 'function' in call
                ])
                print(f"Tool calls: {calls}")
                
            # Ensure we return a string
            return content if content else "Processing your request..."
        elif message["role"] == "tool":
            # For debugging only - not shown to user
            print(f"Tool response: {message.get('content', '')}")
            return None
        return message.get("content", "")
    except Exception as e:
        print(f"Error formatting message: {e}")
        print(f"Problematic message: {message}")
        return "Error processing message"

def run_agent(message: str, history: List[List[str]] = None) -> str:
    if history is None:
        history = []
        
    # Format the history for the agent
    formatted_history = []
    for h in history:
        formatted_history.append({"role": "user", "content": h[0].split("\n")[0]})  # Just use the first line (original message)
        formatted_history.append({"role": "assistant", "content": h[1]})
        
    # Add the current message
    formatted_history.append({"role": "user", "content": message})
    
    # Initialize the agent
    agent = initialize_agent()
    
    # Run the agent
    response = agent_executor.run(agent, formatted_history)
    
    # Process the response messages
    display_messages = []
    response_message = ""
    
    print("FULL RESPONSE MESSAGES:", json.dumps(response.messages, indent=2))
    
    # First, look for tool responses with cryptocurrency price information
    crypto_info = None
    for msg in response.messages:
        if msg.get("role") == "tool" and msg.get("content"):
            try:
                tool_result = json.loads(msg.get("content", "{}"))
                if "crypto" in tool_result and "price_usd" in tool_result and tool_result.get("status") == "success":
                    crypto_info = f"The current price of {tool_result['crypto']} is ${tool_result['price_usd']}."
                    break
            except json.JSONDecodeError:
                pass
    
    # Extract all meaningful content from the agent's response
    for msg in response.messages:
        if msg.get("role") == "assistant":
            if msg.get("content"):
                # Make sure we're not getting a raw function call as content
                content = msg.get("content", "")
                # Skip if it looks like JSON
                if not (content.startswith('{') and content.endswith('}')) and not content.startswith('{"name":'):
                    display_messages.append(content)
    
    # Join all display messages
    response_message = " ".join(display_messages).strip()
    
    # If we found crypto price info but it's not in the response, use it instead
    if crypto_info and (not response_message or '{"name":' in response_message):
        response_message = crypto_info
    
    # If we still don't have a valid response or it contains JSON
    if not response_message or '{' in response_message or '}' in response_message:
        # If we have crypto info, just use that
        if crypto_info:
            response_message = crypto_info
        else:
            # Final fallback if nothing else worked
            response_message = "I'm sorry, I couldn't retrieve the cryptocurrency information you requested. Please try asking in a different way."
    
    print("FINAL DISPLAY:", response_message)
    
    return response_message

def process_conversation(history, user_input):
    print("\n" + "="*50)
    print(f"Processing user input: '{user_input}'")
    
    # Add user message to history
    history.append({"role": "user", "content": user_input})
    print(f"History after adding user message:")
    safe_print(history, "HISTORY:")
    
    try:
        # Run the agent
        print("Calling run_agent()...")
        response = run_agent(user_input, history)
        print("Agent response received:")
        print("RESPONSE:", response)
        
        # Update history with agent responses
        history.append({"role": "assistant", "content": response})
        print("Updated history:")
        safe_print(history, "UPDATED HISTORY:")
        
        # Convert to Gradio chat format
        display_history = []
        
        # Process messages in pairs (user message followed by final response)
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                user_content = history[i]["content"]
                
                # Find the next final assistant response
                final_response = None
                
                # Look ahead for relevant messages
                j = i + 1
                while j < len(history):
                    msg = history[j]
                    
                    # If this is a final assistant response with content, we'll use it
                    if msg["role"] == "assistant" and msg.get("content"):
                        final_response = msg["content"]
                        break
                    
                    j += 1
                
                # If we found a final response
                if final_response:
                    display_history.append((user_content, final_response))
                else:
                    # If no final response found, just add the user message
                    display_history.append((user_content, "Processing your request..."))
            
            # Move to the next user message
            i += 1
            while i < len(history) and history[i]["role"] != "user":
                i += 1
        
        print("Final display history:")
        safe_print(display_history, "DISPLAY HISTORY:")
        print("="*50 + "\n")
        return history, display_history
    except Exception as e:
        print(f"ERROR in conversation: {e}")
        import traceback
        traceback.print_exc()
        # Handle errors gracefully
        error_msg = {"role": "assistant", "content": "I encountered an error. Could you please try again?"}
        history.append(error_msg)
        display_history = []
        for i, msg in enumerate(history):
            if msg["role"] == "user":
                display_history.append((msg.get("content", ""), None))
            elif msg["role"] == "assistant" and msg.get("content"):
                if len(display_history) > 0:
                    display_history[-1] = (display_history[-1][0], msg.get("content", ""))
        
        print("Error display history:")
        safe_print(display_history, "ERROR DISPLAY HISTORY:")
        print("="*50 + "\n")
        return history, display_history

def create_gradio_interface():
    # Define examples in multiple languages
    examples = [
        ["What's the current price of Bitcoin?"],
        ["How much is Ethereum worth right now?"],
        ["What's the price of SOL?"],
        ["Tell me the prices of BTC and ETH"],
        ["Quel est le prix de Cardano?"],  # French
        ["Was kostet Bitcoin jetzt?"],  # German
        ["बिटकॉइन का मूल्य क्या है?"],  # Hindi
        ["比特币现在值多少钱?"]  # Chinese
    ]
    
    # Create a custom theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate"
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",
        background_fill_primary="*neutral_50"
    )
    
    # Define CSS for custom styling
    custom_css = """
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-container h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .title-container p {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    .chat-container {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    footer {
        visibility: hidden;
    }
    """
    
    with gr.Blocks(theme=theme, css=custom_css) as demo:
        with gr.Column():
            # Title and description
            with gr.Row(elem_classes="title-container"):
                with gr.Column():
                    gr.Markdown("# 🌐 Multilingual Cryptocurrency Assistant")
                    gr.Markdown("Ask about cryptocurrency prices in any language! Supports real-time price information for Bitcoin, Ethereum, Solana, and many more.")
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=4, elem_classes="chat-container"):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=450,
                        bubble_full_width=False,
                        show_copy_button=True
                    )
            
            # Input and buttons
            with gr.Row():
                with gr.Column(scale=8):
                    msg = gr.Textbox(
                        placeholder="Ask me about cryptocurrency prices (e.g., 'What's the price of Bitcoin?')...",
                        label="Your Message",
                        scale=8,
                        container=False,
                        show_label=False
                    )
                with gr.Column(scale=1):
                    clear = gr.Button("🧹 Clear")
            
            # Examples
            gr.Examples(
                examples=examples,
                inputs=msg,
                label="Example Queries"
            )
            
            # Information about supported cryptocurrencies
            with gr.Accordion("Supported Cryptocurrencies", open=False):
                gr.Markdown("""
                The assistant supports a wide range of cryptocurrencies, including:
                
                | Symbol | Name | Symbol | Name |
                |--------|------|--------|------|
                | BTC | Bitcoin | ETH | Ethereum |
                | SOL | Solana | DOGE | Dogecoin |
                | XRP | Ripple | ADA | Cardano |
                | DOT | Polkadot | LINK | Chainlink |
                | LTC | Litecoin | AVAX | Avalanche |
                | MATIC | Polygon | SHIB | Shiba Inu |
                | UNI | Uniswap | XLM | Stellar |
                
                You can refer to cryptocurrencies by their full name or symbol.
                """)
            
            # How it works section
            with gr.Accordion("How It Works", open=False):
                gr.Markdown("""
                This assistant uses a combination of technologies to provide cryptocurrency information in multiple languages:
                
                1. **Language Detection**: When you ask a question in any language, the system detects it automatically
                2. **Translation**: Non-English queries are translated to English using advanced AI
                3. **Cryptocurrency Price**: The system fetches real-time price data from CoinGecko
                4. **Response Generation**: A helpful response is crafted based on the price information
                
                All of this happens seamlessly, so you can interact in your preferred language!
                """)
        
        # Logic for message handling
        def respond(message, chat_history):
            if not message:
                return "", chat_history
                
            # Special handling for non-English messages
            try:
                from langdetect import detect
                lang = detect(message)
                if lang != 'en':
                    # Add a note that this is translated
                    display_message = f"{message}\n(Translated: "
                    
                    # Try to translate for display purposes
                    try:
                        from tools import translator
                        translated, _ = translator.translate_to_english(message)
                        display_message += f"{translated})"
                    except Exception:
                        display_message += "Processing non-English query...)"
                        
                    # Use the original message for processing but show translated version
                    bot_response = run_agent(message, chat_history)
                    chat_history.append((display_message, bot_response))
                    return "", chat_history
                else:
                    bot_response = run_agent(message, chat_history)
                    chat_history.append((message, bot_response))
                    return "", chat_history
            except Exception as e:
                # If language detection fails, just process normally
                print(f"Error in respond function: {e}")
                bot_response = run_agent(message, chat_history)
                chat_history.append((message, bot_response))
                return "", chat_history
            
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name=HOST, server_port=PORT, share=SHARE)