from openai import OpenAI
import os
from dotenv import load_dotenv
from guardrails import Guard

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.4
MAX_TOKENS = 2500
STREAM = False

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)

# Load your rail spec
guard = Guard.from_rail("safety_guardrails.rail")

SYSTEM_PROMPT = """You are a polite and helpful robot assistant.  
Rules:  
- Never give unsafe advice.  
- Always respond with a friendly and professional tone.  
- If unsure, say: “I am not certain about that, but I can help you if you give more information.”  
Format your output as JSON with one key: "response".
"""

USER_PROMPT = "this is user prompt"

MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
]

print(response.validated_output)

# Wrap into a helper function
def safe_response(user_input: str):
    return guard(
        openai.chat.completions.create,
        prompt_params={"user_input": user_input},
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=STREAM
    ).validated_output

# Test
print(safe_response("Hi there!"))
print(safe_response("For blood pressure what kind of the drugs do you sugest?"))
