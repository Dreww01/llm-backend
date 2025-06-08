# FastAPI main app

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_engine import agent_executor, parser  # Import agent_executor and parser from llm_engine.py
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://drewgpt.vercel.app/"],  # Or use your frontend URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    input: str

@app.get("/")
def root():
    return {"message": "Hello, Welcome to my LLM"}

@app.post("/chat")
async def generate(prompt: Prompt):
    try:
        raw_response = agent_executor.invoke({"query": prompt.input})
        print("Raw response:", raw_response)
        structured_response = parser.parse(raw_response.get("output"))
        print("Structured response:", structured_response)
        return {"response": structured_response}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
