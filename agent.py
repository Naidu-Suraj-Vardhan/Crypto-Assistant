import json
import time
import os
import inspect
from enum import Enum
from typing import List, Dict, Any, Callable, Optional, Literal, Union
from pydantic import BaseModel
from dotenv import load_dotenv
import together

# Load environment variables
load_dotenv()

# Initialize Together AI client
together.api_key = os.getenv("TOGETHER_API_KEY")

class MessageRole(str, Enum):
    """Enum for message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Function(BaseModel):
    arguments: str
    name: str

class ToolCall(BaseModel):
    function: Function
    id: str
    type: str

class Agent(BaseModel):
    """Agent definition"""
    name: str
    model: str
    instructions: str
    functions: List[Callable] = []
    tool_choice: Optional[str] = "auto"

class AgentResponse(BaseModel):
    messages: List = []

def function_to_json(func: Callable) -> Dict[str, Any]:
    """Convert a function to a Together AI function schema
    
    Args:
        func: The Python function to convert
    
    Returns:
        The function in Together AI schema format
    """
    import inspect
    import re
    from typing import get_type_hints
    
    # Get function signature
    signature = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Extract description from docstring
    description = doc.split("\n\n")[0].strip() if doc else ""
    
    # Parse parameters from docstring
    param_pattern = r"Args:(.*?)(?:Returns:|Raises:|$)"
    param_match = re.search(param_pattern, doc, re.DOTALL)
    param_docs = {}
    if param_match:
        param_section = param_match.group(1)
        param_entries = re.findall(r"(\w+)\s*\((.*?)\):\s*(.*?)(?=\w+\s*\(|\Z)", param_section + "dummy (dummy):", re.DOTALL)
        for name, type_hint, desc in param_entries:
            param_docs[name] = desc.strip()
    
    # Get typing info
    type_hints = get_type_hints(func)
    
    # Build parameters object
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, param in signature.parameters.items():
        # Skip self and kwargs
        if name == "self" or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
            
        # Get parameter type
        param_type = "string"  # Default to string
        if name in type_hints:
            hint = type_hints[name]
            if hint == int:
                param_type = "number"
            elif hint == bool:
                param_type = "boolean"
            elif hint == List[str]:
                param_type = "array"
                
        # Add to properties
        parameters["properties"][name] = {
            "type": param_type,
            "description": param_docs.get(name, "")
        }
        
        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)
    
    # Build function schema
    function_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": parameters
        }
    }
    
    return function_schema

class AgentExecutor:
    """Agent execution engine that runs the agent and calls functions"""
    
    def __init__(self):
        self.client = together.Together()
        
    def get_chat_completion(self, agent: Agent, history: List, model_override: str):
        # Enhance system prompt with chain-of-thought instructions
        system_prompt = agent.instructions
        if "step" not in system_prompt.lower():
            system_prompt += """

When calling tools, follow these steps EXACTLY:
1. Identify IF a tool needs to be called based on the user's query
2. Choose the CORRECT tool and parameters 
3. Call the tool properly through the function calling interface
4. When you receive the tool response, incorporate it into your answer
5. NEVER return raw JSON function calls or function calling syntax to the user

CRITICAL FOR NON-ENGLISH QUERIES:
- FIRST detect if the message is NOT in English
- If NOT in English, ALWAYS call translate_text FIRST before any other actions
- After translation, decide if it's a cryptocurrency query or general conversation

CRITICAL FOR CRYPTO QUERIES:
- ALWAYS call get_crypto_price when the user mentions a cryptocurrency AND is asking about price
- ALWAYS recognize abbreviations like "eth", "btc", "sol" as cryptocurrencies
- Examples of price questions: "what's the price of X", "how much is X worth", "tell me about X price", "check X price"
- For "what's the price of eth?", you MUST call get_crypto_price(crypto="ethereum")

GENERAL CONVERSATION:
- If the user is just saying hello or having a general conversation, respond in a friendly tone
- Don't call any tools for greetings, thanks, or general chit-chat
- Always maintain a helpful and conversational tone

Example thought processes:
"User is asking about bitcoin price, so I need to call get_crypto_price with parameter 'bitcoin'"
"User asked about eth price, so I need to call get_crypto_price with parameter 'ethereum'"
"The message is in Hindi, so I need to first call translate_text"
"User is just saying hello, so I'll respond conversationally without calling any tools"

Bad outputs to AVOID:
{"name": "get_crypto_price", "parameters": {"crypto": "bitcoin"}}
Function: get_crypto_price(crypto="bitcoin")
"The price of eth is $X" (without calling get_crypto_price)
"""

        # Continue with standard completion logic
        messages = [{"role": "system", "content": system_prompt}] + history
        tools = [function_to_json(f) for f in agent.functions]
        
        print("\n" + "="*50)
        print(f"API REQUEST PREPARATION - Model: {model_override or agent.model}")
        
        # Sanitize messages to ensure they are properly formatted
        sanitized_messages = []
        for i, msg in enumerate(messages):
            # Only include valid fields based on role
            clean_msg = {"role": msg["role"]}
            
            # Add content if it exists
            if "content" in msg and msg["content"] is not None:
                clean_msg["content"] = msg["content"]
            else:
                clean_msg["content"] = ""  # Ensure content is never None
                
            # Add tool_calls if they exist and role is assistant
            if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                # Ensure tool_calls are correctly formatted
                formatted_tool_calls = []
                for tool_call in msg["tool_calls"]:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        formatted_tool_calls.append(tool_call)
                if formatted_tool_calls:
                    clean_msg["tool_calls"] = formatted_tool_calls
                
            # Handle tool responses
            if msg["role"] == "tool" and "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
                
            sanitized_messages.append(clean_msg)
            print(f"Message {i} (role={msg['role']}): {clean_msg}")
        
        print(f"Total messages: {len(sanitized_messages)}")
        if tools:
            print(f"Tools provided: {[t['function']['name'] for t in tools]}")
        
        request_params = {
            "model": model_override or agent.model,
            "messages": sanitized_messages,
            "tool_choice": agent.tool_choice,
        }
        
        # Only add tools if they exist
        if tools:
            request_params["tools"] = tools
        
        try:
            print("Making API request to Together AI...")
            response = self.client.chat.completions.create(**request_params)
            print("API response received successfully")
            
            # Print response details for debugging
            try:
                choice = response.choices[0]
                message = choice.message
                print(f"Response message role: {message.role}")
                print(f"Response message content: {message.content}")
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"Tool calls in response: {len(message.tool_calls)}")
                    for tc in message.tool_calls:
                        print(f"  Tool call: {tc.function.name}({tc.function.arguments})")
            except Exception as e:
                print(f"Error printing response details: {e}")
            
            print("="*50 + "\n")
            return response
        except Exception as e:
            print(f"API ERROR: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            print("="*50 + "\n")
            # Create a fallback response
            from types import SimpleNamespace
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            model_dump=lambda: {
                                "role": "assistant",
                                "content": "I'm sorry, I encountered an error processing your request. Could you please try again?",
                                "tool_calls": []
                            }
                        )
                    )
                ]
            )

    def handle_function_result(self, result) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result
        elif isinstance(result, Agent):
            return {"content": json.dumps({"assistant": result.name})}
        try:
            return {"content": str(result)}
        except Exception as e:
            raise TypeError(f"Could not convert result to string: {e}")

    def handle_tool_calls(self, tool_calls: List[ToolCall], functions: List[Callable]) -> AgentResponse:
        function_map = {f.__name__: f for f in functions}
        response = AgentResponse(messages=[])
        
        for tool_call in tool_calls:
            name = tool_call.function.name
            if name not in function_map:
                response.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": f"Error: Tool {name} not found"})
                continue
                
            args = json.loads(tool_call.function.arguments)
            result = function_map[name](**args)
            processed = self.handle_function_result(result)
            
            response.messages.append(processed)
            if processed.get("content"):
                response.messages.append({"role": "assistant", "content": processed["content"]})

        return response

    def run(self, agent: Agent, messages: List, max_turns=10):
        """
        Run the entire agent conversation flow
        
        Args:
            agent: The agent definition
            messages: The conversation history
            max_turns: Maximum number of agent turns to take

        Returns:
            AgentResponse object with the final messages
        """
        current_messages = messages.copy()
        new_messages = []
        
        # Check if the latest user message is in a non-English language
        if messages and messages[-1]["role"] == "user":
            user_message = messages[-1]["content"]
            try:
                from langdetect import detect
                lang = detect(user_message)
                print(f"Detected language: {lang}")
                
                # If non-English, translate first
                if lang != 'en':
                    print("Non-English message detected, translating first...")
                    from tool import translate_text
                    
                    # Create translation tool call
                    translation_call = {
                        "role": "assistant", 
                        "content": None, 
                        "tool_calls": [
                            {
                                "id": f"auto_translation_{int(time.time())}", 
                                "type": "function", 
                                "function": {
                                    "name": "translate_text", 
                                    "arguments": json.dumps({"text": user_message})
                                }
                            }
                        ]
                    }
                    
                    # Execute translation and add to messages
                    translation_result = translate_text(text=user_message)
                    try:
                        translation_data = json.loads(translation_result)
                        translated_text = translation_data.get("translated_text", "")
                        detected_language = translation_data.get("detected_language", "unknown")
                        
                        # Add these messages to the history
                        new_messages.append(translation_call)
                        translation_response = {
                            "role": "tool", 
                            "tool_call_id": translation_call["tool_calls"][0]["id"], 
                            "content": translation_result
                        }
                        new_messages.append(translation_response)
                        
                        # Add a system message to explain what happened
                        new_messages.append({
                            "role": "system", 
                            "content": f"The user's message was in {detected_language} and has been translated to: '{translated_text}'. Please respond to this translated query now."
                        })
                        
                        # Update current messages
                        current_messages.extend(new_messages)
                        
                        print(f"Translated '{user_message}' to '{translated_text}'")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse translation result: {e}")
            except Exception as e:
                print(f"Language detection error: {e}")
                # Continue with normal processing if language detection fails
        
        # Main agent conversation loop
        remaining_turns = max_turns
        while remaining_turns > 0:
            remaining_turns -= 1
            print(f"Agent turn {max_turns - remaining_turns}/{max_turns}")
            
            # Get the next agent response
            completion = self.get_chat_completion(agent, current_messages, None)
            if not completion or not hasattr(completion, 'choices') or not completion.choices:
                print("Error: Invalid completion response")
                break
                
            choice = completion.choices[0]
            if not hasattr(choice, 'message'):
                print("Error: No message in completion choice")
                break
                
            # Convert message to a dictionary
            try:
                assistant_message = choice.message.model_dump()
            except AttributeError:
                # Handle case where model_dump isn't available
                assistant_message = {
                    "role": "assistant",
                    "content": choice.message.content if hasattr(choice.message, 'content') else "",
                }
                if hasattr(choice.message, 'tool_calls'):
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in choice.message.tool_calls
                    ]
            
            # Add the assistant's response to the conversation
            new_messages.append(assistant_message)
            current_messages.append(assistant_message)
            
            # Check if the assistant wants to use a tool
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                # No more tool calls, we're done
                break
                
            # Process each tool call
            for tool_call in tool_calls:
                # Make sure the tool_call has the expected structure
                if isinstance(tool_call, dict) and "function" in tool_call:
                    function_name = tool_call["function"]["name"]
                    try:
                        function_args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        function_args = {}
                    function_call_id = tool_call["id"]
                    
                    # Find the matching function
                    target_function = None
                    for function in agent.functions:
                        if function.__name__ == function_name:
                            target_function = function
                            break
                    
                    if target_function:
                        # Execute the function
                        try:
                            print(f"Executing tool call: {function_name}({function_args})")
                            function_result = target_function(**function_args)
                            
                            # Parse the result to extract the actual response message
                            try:
                                parsed_result = json.loads(function_result)
                                # If this is a successful response with a message field, use it
                                if "message" in parsed_result:
                                    # Store both the raw result and the display message
                                    function_message = {
                                        "role": "tool",
                                        "tool_call_id": function_call_id,
                                        "content": function_result,
                                        "display_message": parsed_result["message"]
                                    }
                                else:
                                    function_message = {
                                        "role": "tool",
                                        "tool_call_id": function_call_id,
                                        "content": function_result
                                    }
                            except (json.JSONDecodeError, TypeError):
                                # Not JSON, just use as is
                                function_message = {
                                    "role": "tool",
                                    "tool_call_id": function_call_id,
                                    "content": function_result
                                }
                                
                        except Exception as e:
                            # Handle function execution errors
                            error_message = f"Error executing {function_name}: {str(e)}"
                            function_message = {
                                "role": "tool",
                                "tool_call_id": function_call_id,
                                "content": json.dumps({"status": "error", "message": error_message})
                            }
                    else:
                        # Function not found
                        error_message = f"Function {function_name} not found"
                        function_message = {
                            "role": "tool",
                            "tool_call_id": function_call_id,
                            "content": json.dumps({"status": "error", "message": error_message})
                        }
                    
                    # Add the function result to the conversation
                    new_messages.append(function_message)
                    current_messages.append(function_message)
                else:
                    print(f"Invalid tool call format: {tool_call}")
            
            # Let the agent continue if it might need more tool calls
            completion = self.get_chat_completion(agent, current_messages, None)
            if not completion or not hasattr(completion, 'choices') or not completion.choices:
                print("Error: Invalid final completion response")
                break
                
            choice = completion.choices[0]
            if not hasattr(choice, 'message'):
                print("Error: No message in final completion choice")
                break
                
            # Convert final message to a dictionary
            try:
                final_message = choice.message.model_dump()
            except AttributeError:
                # Handle case where model_dump isn't available
                final_message = {
                    "role": "assistant", 
                    "content": choice.message.content if hasattr(choice.message, 'content') else ""
                }
                if hasattr(choice.message, 'tool_calls'):
                    final_message["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in choice.message.tool_calls
                    ]
            
            # If the final message has more tool calls, continue the loop
            if final_message.get("tool_calls"):
                current_messages.append(final_message)
                new_messages.append(final_message)
                continue
                
            # Ensure we have a final message with content
            if not final_message.get("content"):
                # Force a final message with content
                current_messages.append({
                    "role": "system",
                    "content": "Please provide a final response based on the tool results above. NEVER call any more tools."
                })
                completion = self.get_chat_completion(agent, current_messages, None)
                if completion and hasattr(completion, 'choices') and completion.choices:
                    choice = completion.choices[0]
                    if hasattr(choice, 'message'):
                        try:
                            final_message = choice.message.model_dump()
                        except AttributeError:
                            final_message = {
                                "role": "assistant",
                                "content": choice.message.content if hasattr(choice.message, 'content') else "I'm sorry, I encountered an error processing your request."
                            }
            
            new_messages.append(final_message)
            break  # We've completed processing, so we're done
                
        return AgentResponse(messages=new_messages)
