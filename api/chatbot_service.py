import os
import json
import re
from typing import Dict, Optional
from groq import Groq


class ChurnChatbotService:
    """Service for handling chatbot conversations with Groq AI"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"  
        self.conversation_history = []
        
        # System prompt for the chatbot
        self.system_prompt = """You are a helpful AI assistant for a telecom company's customer churn prediction system.

Your role is to:
1. Have friendly conversations with users about customer information
2. Extract customer details from natural language
3. Ask clarifying questions if information is missing
4. Format data correctly for the prediction API

Required customer information:
- customerID (optional, can be generated)
- gender (Male/Female)
- Senior_Citizen (0 or 1, ask if customer is senior/elderly)
- Is_Married (Yes/No, ask about marital status)
- Dependents (Yes/No)
- tenure (months with company)
- Phone_Service (Yes/No)
- Dual (Yes/No, means multiple phone lines)
- Internet_Service (DSL/Fiber optic/No)
- Online_Security (Yes/No/No internet service)
- Online_Backup (Yes/No/No internet service)
- Device_Protection (Yes/No/No internet service)
- Tech_Support (Yes/No/No internet service)
- Streaming_TV (Yes/No/No internet service)
- Streaming_Movies (Yes/No/No internet service)
- Contract (Month-to-month/One year/Two year)
- Paperless_Billing (Yes/No)
- Payment_Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))
- Monthly_Charges (dollar amount)
- Total_Charges (dollar amount)

When you have enough information, respond with:
ACTION: PREDICT
DATA: {json formatted customer data}

Be conversational, friendly, and ask one question at a time. Make reasonable assumptions when appropriate (e.g., if someone has fiber internet, they probably have internet service).
"""
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def chat(self, user_message: str) -> Dict:
        """
        Process user message and return response
        
        Returns:
        - response: str - The chatbot's response
        - action: str - 'chat' or 'predict'
        - data: dict - Customer data if action is 'predict'
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages for Groq
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]
        
        # Call Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
            )
            
            assistant_message = chat_completion.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Check if the assistant wants to make a prediction
            if "ACTION: PREDICT" in assistant_message:
                return self._extract_prediction_data(assistant_message)
            else:
                return {
                    "response": assistant_message,
                    "action": "chat",
                    "data": None
                }
                
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "action": "error",
                "data": None
            }
    
    def _extract_prediction_data(self, assistant_message: str) -> Dict:
        """Extract customer data from assistant's response"""
        try:
            # Find JSON data in the response
            json_match = re.search(r'DATA:\s*({.*?})', assistant_message, re.DOTALL)
            
            if json_match:
                customer_data = json.loads(json_match.group(1))
                
                # Clean the response (remove ACTION and DATA parts)
                clean_response = re.sub(r'ACTION:.*?DATA:.*', '', assistant_message, flags=re.DOTALL).strip()
                
                if not clean_response:
                    clean_response = "I've gathered all the information. Let me check the churn prediction for this customer..."
                
                return {
                    "response": clean_response,
                    "action": "predict",
                    "data": customer_data
                }
            else:
                return {
                    "response": assistant_message,
                    "action": "chat",
                    "data": None
                }
                
        except json.JSONDecodeError:
            return {
                "response": "I have the information but encountered an error formatting it. Could you please provide the details again?",
                "action": "error",
                "data": None
            }