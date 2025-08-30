import os
from dotenv import load_dotenv
import json
import google.generativeai as genai

MODEL="gemini-2.5-flash"

IMAGE_PATH="images/images.jpeg"
with open(IMAGE_PATH, "rb") as image_file:
    image_data = image_file.read()



TEMPERATURE=0.4
MAX_TOKENS=2500

load_dotenv(override=True)
google_api_key=os.getenv("GOOGLE_API_KEY")
gemini=genai.configure(api_key=google_api_key)
model = genai.GenerativeModel(MODEL)

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

{{
  "selection": {{
    "reason": "short explanation of why this object was chosen",
    "object_id": "oX"  // id from the objects list, or null if none fits
  }},
  "action": {{
    "type": "pick_and_place|hand_over|point_at|clarify",
    "target_object_id": "oX",
    "destination": "to_user|cart|left_side|right_side|stay|unknown"
  }},
  "safety": {{
    "need_clarification": true|false,
    "message_if_any": "a short question to ask the user if clarification is needed"
  }},
  "spoken_reply": "a friendly one-sentence reply to the user"
}}

Constraints:
- Match user request to objects by color, name, or attributes.  
- If multiple objects match, set need_clarification=true.  
- The spoken_reply should be polite and natural.
"""


#Perception = sense the environment → objects
#Grounding = link user intent to objects → action



per_response = model.generate_content([
    PER_SYSTEM_PROMPT + "\n\n" + PER_USER_PROMPT,
    { "mime_type": "image/jpeg", "data": image_data }
])


print(per_response.text)


try:
    per_objects = json.loads(per_response.text)
except json.JSONDecodeError:
    print("Raw perception output:", per_response.text)
    per_objects = {}

gr_user_prompt = GR_USER_PROMPT.format(
    objects_json=json.dumps(per_objects, indent=2),
    query="hand me the cake"
)


gr_response = model.generate_content([
    GR_SYSTEM_PROMPT,
    gr_user_prompt],
    generation_config={"temperature":TEMPERATURE, "max_output_tokens":MAX_TOKENS}
    )

try:
    gr_objects = json.loads(gr_response.text)
    print("Grounding JSON:", json.dumps(gr_objects, indent=2))
except json.JSONDecodeError:
    print("Raw grounding output:", gr_response.text)


print("Perception JSON:", json.dumps(per_objects, indent=2))
print("Grounding JSON:", gr_response.text)

if "gr_objects" in locals() and "spoken_reply" in gr_objects:
    print("Robot says:", gr_objects["spoken_reply"])
else:
    print("Robot says: (could not parse spoken_reply)")
