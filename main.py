from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import hashlib
import time
import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from groq import Groq

# ---- SETUP ----
os.environ.setdefault("GROQ_API_KEY", "")

app_fastapi = FastAPI()

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("fire_model_final.h5")
class_names = ['fire', 'nofire', 'smoke']
client = Groq()
ledger = []

# ---- AGENT STATE ----
class AgentState(TypedDict):
    image_array: object
    prediction: Optional[str]
    confidence: Optional[float]
    decision: Optional[str]
    alert_level: Optional[str]
    action_log: Optional[str]
    ledger_hash: Optional[str]
    reasoning: Optional[str]

# ---- OBSERVE ----
def observe(state: AgentState) -> AgentState:
    preds = model.predict(state['image_array'], verbose=0)[0]
    pred_idx = np.argmax(preds)
    state['prediction'] = class_names[pred_idx]
    state['confidence'] = float(preds[pred_idx])
    return state

# ---- DECIDE ----
import re

def decide(state: AgentState) -> AgentState:
    pred = state['prediction']
    conf = state['confidence']

    prompt = f"""You are a satellite fire-monitoring AI agent making a safety decision.

Detection result: "{pred}" with {conf*100:.1f}% confidence.

Rules of thumb:
- High confidence (>85%) fire/smoke = urgent, low confidence = needs human verification
- nofire is always safe unless confidence is very low

Respond in EXACTLY this format, nothing else:
DECISION: <one short sentence>
ALERT_LEVEL: <none/low/medium/critical>
REASONING: <one sentence explaining why>
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = response.choices[0].message.content

    decision_match = re.search(r"DECISION:\s*(.*?)(?=ALERT_LEVEL:|$)", text, re.DOTALL)
    alert_match = re.search(r"ALERT_LEVEL:\s*(.*?)(?=REASONING:|$)", text, re.DOTALL)
    reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.DOTALL)

    state['decision'] = decision_match.group(1).strip() if decision_match else "unknown"
    state['alert_level'] = alert_match.group(1).strip().lower() if alert_match else "medium"
    state['reasoning'] = reasoning_match.group(1).strip() if reasoning_match else ""

    return state

# ---- ACT ----
def act(state: AgentState) -> AgentState:
    prev_hash = ledger[-1]['hash'] if ledger else "0"*64
    entry_data = f"{state['prediction']}|{state['confidence']}|{state['decision']}|{time.time()}|{prev_hash}"
    entry_hash = hashlib.sha256(entry_data.encode()).hexdigest()
    ledger.append({"data": entry_data, "hash": entry_hash})

    state['ledger_hash'] = entry_hash
    state['action_log'] = f"[{state['alert_level'].upper()}] {state['decision']}"
    return state

# ---- BUILD GRAPH ----
graph = StateGraph(AgentState)
graph.add_node("observe", observe)
graph.add_node("decide", decide)
graph.add_node("act", act)
graph.set_entry_point("observe")
graph.add_edge("observe", "decide")
graph.add_edge("decide", "act")
graph.add_edge("act", END)
agent = graph.compile()

# ---- API ENDPOINT ----
@app_fastapi.get("/")
def home():
    return {"message": "Fire Detection Agent API is running"}

@app_fastapi.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)

    result = agent.invoke({"image_array": img_array})

    return {
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "decision": result['decision'],
        "alert_level": result['alert_level'],
        "reasoning": result['reasoning'],
        "ledger_hash": result['ledger_hash']
    }