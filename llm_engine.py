# LLM logic here

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# Specify all the fields you want as output from your llm call.
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Setup LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            After completing your research, always use the 'save_text_to_file' tool to save your results.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Agent and executor
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[],  # No tools for now
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[],
    memory=None,
    verbose=False,
)




