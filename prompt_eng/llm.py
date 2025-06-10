
# == llm.py ==
"""OpenAI Function‑Calling 래퍼."""
import os, json, yaml, pathlib
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

BASE = pathlib.Path(__file__).resolve().parent
RES  = BASE / "resources"
CFG  = yaml.safe_load((RES / "config.yaml").read_text("utf-8"))

MODEL, SYSTEM, USER_TMPL = CFG["model"], CFG["system_prompt"], CFG["user_prompt"]
SCHEMA = yaml.safe_load(CFG["schema"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch(topic: str, pages: int = 2) -> dict:
    """LLM → JSON 구조."""
    user = USER_TMPL.format(topic=topic, pages=pages)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        functions=[SCHEMA],
        function_call={"name": SCHEMA['name']},
        temperature=0.3,
    )
    return json.loads(resp.choices[0].message.function_call.arguments)
