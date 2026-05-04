"""Smoke test for the Eurecom LiteLLM gateway.

Run from the project root:
    python task2_setup_rag/llm/test_eurecom.py

Reads EURECOM_LLM_URL, EURECOM_LLM_KEY, EURECOM_LLM_MODEL from .env.
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = os.getenv("EURECOM_LLM_URL")
API_KEY  = os.getenv("EURECOM_LLM_KEY")
MODEL    = os.getenv("EURECOM_LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")

print("─── Config ───")
print(f"BASE_URL: {BASE_URL}")
print(f"API_KEY : {API_KEY[:10]}…{API_KEY[-4:]}" if API_KEY else "API_KEY : (missing)")
print(f"MODEL   : {MODEL}")
print()

if not (BASE_URL and API_KEY):
    print("Missing EURECOM_LLM_URL or EURECOM_LLM_KEY in .env — aborting.")
    sys.exit(1)


# ── 1. List available models ───────────────────────────────────────────────────
print("─── GET /v1/models ───")
import httpx
try:
    r = httpx.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=10.0)
    print(f"HTTP {r.status_code}")
    if r.status_code == 200:
        for m in r.json().get("data", []):
            mark = "  ←  selected" if m["id"] == MODEL else ""
            print(f"  - {m['id']}{mark}")
    else:
        print(r.text)
except Exception as e:
    print(f"ERROR: {e}")
print()


# ── 2. Chat completion call ────────────────────────────────────────────────────
print("─── POST /v1/chat/completions ───")
from openai import OpenAI, APIError

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
prompt = "What is the capital of France? Answer in one short sentence."
print(f"prompt: {prompt}")
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
    )
    print(f"HTTP 200 — model={response.model}")
    print(f"answer: {response.choices[0].message.content.strip()}")
    usage = getattr(response, "usage", None)
    if usage:
        print(f"tokens: prompt={usage.prompt_tokens}  completion={usage.completion_tokens}  total={usage.total_tokens}")
except APIError as e:
    print(f"APIError {e.status_code}: {e.message}")
    if hasattr(e, "body") and e.body:
        print(f"body: {e.body}")
except Exception as e:
    print(f"ERROR ({type(e).__name__}): {e}")
