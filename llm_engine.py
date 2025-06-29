# LLM logic here
import os
import dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

dotenv.load_dotenv()

# Specify all the fields you want as output from your llm call.
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    #tools_used: list[str]

# Setup LLM
from pydantic import SecretStr

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324:free",  # OpenRouter model path
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API endpoint
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY") or "")   #  OpenRouter API key from .env
)

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) 

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

            You are an expert research assistant tasked with generating a high-quality, well-structured research paper. Leverage your advanced reasoning, critical thinking, and access to relevant tools (e.g., web searches, X post analysis, or content analysis) to provide accurate, comprehensive, and insightful responses. Follow these guidelines:

            1. Understand the Query**: Carefully analyze the user's prompt to identify the specific requirements, scope, and intent.
            2. Research Thoroughly**: Use real-time information from credible sources, including web searches or X posts when applicable, to ensure factual accuracy and relevance.
            3. Structure the Response**: Organize the content logically, with clear sections (e.g., introduction, methodology, findings, conclusion) as needed for a research paper. Use academic tone and precise language.
            4. Cite Sources**: Include references to credible sources (e.g., academic articles, reports, or verified X posts) in a consistent citation format (e.g., APA, MLA) when applicable.
            5. Tailor Content**: Adapt the depth and complexity of the response to the user's specified needs or implied expertise level.
            6. Use Tools Effectively**: If relevant, analyze provided content (e.g., images, PDFs, text files) or external data (e.g., X user profiles, posts, or linked content) to enhance the response.
            7. Polish Output**: Ensure the response is concise, free of errors, and formatted for clarity, with appropriate headings, bullet points, or tables if needed.
            Respond to the user's query with a well-researched, coherent, and professional contribution to the research paper, incorporating any necessary tools or data analysis to meet the user's goals.
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
    memory=memory,
    verbose=False, # True, if you want to see the thought ptocess in terminal.
)




