import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from openai import OpenAI
from os import getenv
import re
import pytz

def setup_logger(name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

logger = setup_logger("AI", "logs/ai.log")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)

def vision(messages, model=None):
    vision_client = OpenAI()
    completion = vision_client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content

def ai_request(messages, model, temperature=0.7):
    logger.info(f"Sending request to AI model: {model}")
    try:
        completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        logger.info("Received response from AI")
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in AI request: {str(e)}")
        raise

def completion(messages, model=None):
    return ai_request(messages, model)

def instruct(message, model=None, temperature=0.7):
    return ai_request([{"role": "user", "content": message}], model, temperature=0.7)
