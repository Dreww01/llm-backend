# FastAPI main app

from fastapi import FastAPI
from pydantic import BaseModel
from llm_engine import agent_executor, parser  # Import agent_executor and parser from llm_engine.py
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use your frontend URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    input: str

@app.get("/")
def root():
    return {"message": "Hello, Welcome to my LLM"}

@app.post("/generate")
async def generate(prompt: Prompt):
    # Use the agent_executor to process the input and parser to format the output
    try:
        raw_response = agent_executor.invoke({"query": prompt.input})
        structured_response = parser.parse(raw_response.get("output"))
        return {"output": structured_response}
    except Exception as e:
        return {"error": str(e)}
