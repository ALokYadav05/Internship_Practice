from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.getenv('BASE_URL')  # use to load secret url
groq_api_key = os.getenv('GROQ_API') # use to load secret API-keys


# llm = ChatOllama(
#             model='llama3.2:latest',
#             temperature=0.7,
#             base_url=base_url
#         )

# defining the model
llm = ChatGroq(
    api_key = groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

response = llm.invoke([
                HumanMessage(content='Explain what is the job of AI Engineer'),
               # Defining that this message came  directly from human not from system prompt or AI itself
            ])
print(response.content)
