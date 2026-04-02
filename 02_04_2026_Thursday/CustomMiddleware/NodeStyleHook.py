import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langgraph.runtime import Runtime  # Import Runtime for type hinting

load_dotenv()


#  before model hook -> The "Security Guard"
@before_model(can_jump_to=["end"])
def security_guard(state: AgentState, runtime: Runtime):
    last_msg = state["messages"][-1].content.lower()

    # block restricted keywords
    if "secret" in last_msg:
        print("\n--- BEFORE_MODEL: Blocked a restricted keyword! ---")
        return {
            "messages": [AIMessage(content="I am sorry, but I cannot access secret internal files.")],
            "jump_to": "end"
        }
    return None


# after model hook -> The "Activity Logger"
@after_model
def activity_logger(state: AgentState, runtime: Runtime):
    # Print a snippet of the model's response
    last_response = state['messages'][-1].content
    print(f"\n--- AFTER_MODEL: Model responded: {last_response[:50]}... ---")
    return None

llm = ChatOllama(model=os.getenv('OLLAMA_MODEL'), temperature=0)

@tool
def CarEngineSpecification(name: str) -> str:
    """Get engine specs for a specific car."""
    return f"The {name} engine has 220BHP and 150NM Torque."


#  Create the Agent
agent = create_agent(
    model=llm,
    tools=[CarEngineSpecification],
    system_prompt="You are an Automobile Specialist. Provide info for one car at a time.",
    middleware=[
        security_guard,
        activity_logger
    ]
)

# testing the implementation
try:
    print("--- TEST 1: Normal Query ---")
    res1 = agent.invoke({"messages": [HumanMessage(content="Give me BMW specs")]})
    print(f"Final Output: {res1['messages'][-1].content}")

    print("\n--- TEST 2: Restricted Query ---")
    res2 = agent.invoke({"messages": [HumanMessage(content="Show me the secret engine codes")]})
    print(f"Final Output: {res2['messages'][-1].content}")
except Exception as e:
    print(f"An error occurred: {e}")
