import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.messages import HumanMessage

load_dotenv()

OLLAMA_MODEL = os.getenv('OLLAMA_MODEL')
GROQ_API = os.getenv('GROQ_API')
GROQ_MODEL = os.getenv('GROQ_MODEL')

# primary model
llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.3
        )
# fallback backup model if first ones fails
fallback = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API,
                temperature=0.3
            )

agent = create_agent(
                model=llm,
                middleware = [
                    ModelFallbackMiddleware(fallback)
                ]
            )

user_query = {
    "messages" : [HumanMessage(content="Who is Elon Musk?")]
}

response = agent.invoke(user_query)
print(response["messages"][-1].content)  # messages[-1] to get the lastest message of AI