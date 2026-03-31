import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
checkPointer = MemorySaver()

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3
)

agent = create_agent(
    model=llm,
    checkpointer=checkPointer,
    middleware=[
        SummarizationMiddleware(
             model=llm,
             trigger=('messages',5),
             keep=("messages",5)
        )
    ]
)

config = {
    "configurable" : {
        "thread_id":"1"
    }
}

print("\n\n ### ChatBot with Memory ('exit' to quit) ### \n")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break

    input_user = {
                    "messages":[
                           HumanMessage(content=user_input)
                    ]
                }
    result = agent.invoke(input_user,config=config)
    print("AI : ", result["messages"][-1].content)



