import os
from dotenv import load_dotenv
import time
from langchain_ollama import ChatOllama
from langchain.agents.middleware import wrap_model_call
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

llm = ChatOllama(
            model = OLLAMA_MODEL,
            temperature=0.3
        )

"""
------- Wrap-style hook middleware --------
-> These wrap around the entire action (like the model call). 
-> You receive a handler (the actual function that calls the model), and you decide if and when to call it.
Example (Security Guard): A guard stands at the door. They can stop you, ask for ID, or even tell you to come back three 
times (retry) before letting you in.
Use Cases: Retries (if the model fails, call it again), 
Caching (if you have the answer, don't call the model at all), or switching models dynamically"""

@wrap_model_call
def timing_middleware(request, handler):
    print("\n Before Model call")
    start = time.time()

    response = handler(request)
    end = time.time()
    print("\n After Model call")
    print(f"Execution time: {end - start}")

    if response.result and len(response.result) > 0:
        response.result[0].content += "\n\n[Processed through middleware]"

    return response

agent = create_agent(
                model=llm,
                tools=[],
                middleware=[timing_middleware]
            )

result = agent.invoke({ "messages": [HumanMessage(content="what is langgraph?")] })

for msg in result['messages']:
    print(f"\n{msg.type.upper()} : {msg.content}")