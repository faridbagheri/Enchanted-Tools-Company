from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import base64

MODEL="gpt-4o-mini"

IMAGE_PATH="images/table_image.jpg"
with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")


TEMPERATURE=0.4
MAX_TOKENS=2500
RESPONSE_FORMAT={"type": "json_object"}
STREAM=False

load_dotenv(override=True)
openai_api_key=os.getenv("OPENAI_API_KEY")
google_api_key=os.getenv("GOOGLE_API_KEY")


openai=OpenAI(api_key=openai_api_key)

PER_SYSTEM_PROMPT = """
You are a careful vision-to-JSON annotator.  
Your task is to look at a scene image and output only a strict JSON object.  
The JSON must contain a list of portable tabletop objects you detect, with their attributes.  
Do not include explanations or extra text outside of JSON.
"""
PER_USER_PROMPT = """Analyze this scene and return the objects as JSON with the schema:
{
  "objects": [
    {
      "id": "o1",              // short unique id
      "name": "cup|tray|plate|bottle|badge|utensil|other",
      "color": "blue|red|green|white|black|silver|transparent|other",
      "bbox": {"x": <0..1>, "y": <0..1>, "w": <0..1>, "h": <0..1>}, // normalized coordinates
      "confidence": 0.0-1.0
    }
  ],
  "notes": "very short rationale"
}

Constraints:
- Coordinates must be normalized (0..1 range).  
- If unsure, make a best guess and lower the confidence.  
- Only include small/portable objects, not the table or walls.
"""

GR_SYSTEM_PROMPT = """You convert natural language commands into structured robot actions.  
Always select the best matching object from a provided JSON list and output a strict JSON object.  
If the request is ambiguous, propose a clarifying question inside the JSON.  
Do not include any extra explanation outside of JSON.
"""

GR_USER_PROMPT = """Objects detected in the scene:
{objects_json}

User request:
"{query}"

Return JSON with the schema:

{
  "selection": {
    "reason": "short explanation of why this object was chosen",
    "object_id": "oX"  // id from the objects list, or null if none fits
  },
  "action": {
    "type": "pick_and_place|hand_over|point_at|clarify",
    "target_object_id": "oX",
    "destination": "to_user|cart|left_side|right_side|stay|unknown"
  },
  "safety": {
    "need_clarification": true|false,
    "message_if_any": "a short question to ask the user if clarification is needed"
  },
  "spoken_reply": "a friendly one-sentence reply to the user"
}

Constraints:
- Match user request to objects by color, name, or attributes.  
- If multiple objects match, set need_clarification=true.  
- The spoken_reply should be polite and natural.
"""


#Perception = sense the environment → objects
#Grounding = link user intent to objects → action

# Call 1: Perception
MESSAGES_PER = [
    {"role": "system", "content": PER_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
        {"type": "text", "text": PER_USER_PROMPT}
    ]}
]

per_response = openai.chat.completions.create(
    model=MODEL,
    messages=MESSAGES_PER,
    response_format=RESPONSE_FORMAT,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    stream=STREAM
)

print(per_response.choices[0].message.content)


try:
    per_objects = json.loads(per_response.choices[0].message.content)
except json.JSONDecodeError:
    print("Raw perception output:", per_response.choices[0].message.content)
    per_objects = {}

gr_user_prompt = GR_USER_PROMPT.format(
    objects_json=json.dumps(per_objects, indent=2),
    query="hand me the white cup"
)
# Call 2: Grounding
MESSAGES_GR = [
    {"role": "system", "content": GR_SYSTEM_PROMPT},
    {"role": "user", "content": gr_user_prompt} # includes objects JSON + query
]


gr_response = openai.chat.completions.create(
    model=MODEL,
    messages=MESSAGES_GR,
    response_format=RESPONSE_FORMAT,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    stream=STREAM
)

try:
    gr_objects = json.loads(gr_response.choices[0].message.content)
    print("Grounding JSON:", json.dumps(gr_objects, indent=2))
except json.JSONDecodeError:
    print("Raw grounding output:", gr_response.choices[0].message.content)


print("Perception JSON:", json.dumps(per_objects, indent=2))
print("Grounding JSON:", gr_response.choices[0].message.content)
if "gr_objects" in locals() and "spoken_reply" in gr_objects:
    print("Robot says:", gr_objects["spoken_reply"])
else:
    print("Robot says: (could not parse spoken_reply)")
