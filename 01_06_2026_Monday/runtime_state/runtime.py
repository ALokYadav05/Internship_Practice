from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Context:
    UserName: str

@dynamic_prompt
def inject_user_name(request: ModelRequest) -> str:
    # Safely pull the dynamic variable from the runtime memory
    user_name = request.runtime.context.UserName
    return f"You are a helpful assistant. The user's name is {user_name}."

llm = ChatOllama(
    model=os.getenv("GEMMA_CLOUD_MODEL"),
    temperature=0.3
)

agent = create_agent(
    model=llm,
    middleware=[inject_user_name],
    context_schema=Context  # type: ignore
)

messages={"messages": [HumanMessage(content="What's my name?")]}

result = agent.invoke(messages, context=Context(UserName="Alok Yadav"))  #type: ignore
for message in result["messages"]:
    print(message.content)