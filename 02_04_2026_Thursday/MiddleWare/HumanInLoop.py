import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

groq_api = os.getenv('GROQ_API')
groq_model = os.getenv('GROQ_MODEL')

llm = ChatGroq(
    model=groq_model,
    api_key=groq_api,
    temperature=0.3
)

system_prompt = """ You are a financial assistant.
                    Rules:
                    1. ALWAYS use tools when the query involves stock prices or buying shares.
                    2. NEVER answer from your own knowledge for these actions.
                    3. You must call the appropriate tool:
                       - check_share_price → for price queries
                       - buy_share → for purchase requests
                    4. Return ONLY the tool call when required.
                """
@tool(
    'check_share_price',
    description="This tool is used to check share price"
)
def check_share_price(stock:str) -> str:
    """Checks share price"""
    return f"The current price of the {stock} is $23.5"

@tool(
    'buy_share',
    description="This tool is used to buy shares"
)
def buy_share(share:str) -> str:
    """Buy shares"""
    return f"Successfully bought 5 quantity of {share}"

agent = create_agent(
            model=llm,
            tools=[check_share_price, buy_share],
            checkpointer=InMemorySaver(),
            system_prompt=system_prompt,
            middleware=[
                HumanInTheLoopMiddleware(    # defining human-in-loop middleware
                    interrupt_on = {
                        "buy_share": {
                               "allowed_decisions":['approve','edit','reject'],
                        },
                        "check_share_price":False,
                    }
                )
            ]
        )

config = {"configurable":{"thread_id":"1"}}

while True:
    user_query = input("\n ### Agent (type exit to close) ### \n")
    if user_query == "exit":
        break

    # running the agent
    state = agent.invoke(
        {"messages": [HumanMessage(content=user_query)]},
              config=config
    )
    print(state)

    if hasattr(state, "interrupts") and state.interrupts:
        print(f"\n--- INTERRUPT: Approval needed for {state.interrupts[0].action_request.name} ---")

        # Only resume if there was an interrupt
        state = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )

        # Print the final message from the assistant
    print(f"\nResponse: {state['messages'][-1].content}\n")

