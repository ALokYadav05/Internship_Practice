import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
    verbose=True
)

system_prompt = """
                   You are helpful Automobile Specialist.
                   Give one car specification at a time.
                   Call tools separately in different steps, if user asks for multiple information.
                   Do not combine tool calls
                """

@tool(
    'CarEngineSpecification',
    description="This tool is used to give Engine specification of specific mentioned car"
)
def CarEngineSpecification(name:str)->str:
    return f"This {name} is very Powerful. It has 220BHP and 150NM Torque"

agent = create_agent(
            model = llm,
            tools = [CarEngineSpecification],
            system_prompt= system_prompt,
            middleware = [
                ModelCallLimitMiddleware(
                    run_limit=1,
                    exit_behavior='error'
                )
            ]
        )

try:
    user_query = {"messages": [HumanMessage(content="Give me details about BMW and Mercedes Engine")]}
    response = agent.invoke(user_query)
    print(response['messages'][-1].content)
except Exception as e:
    print(f"\n Successfully triggered Error: {e}")
