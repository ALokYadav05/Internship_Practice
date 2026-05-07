from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

def get_api_key():
    """
    This method is used to get api key from dotenv
    :return: groq api key
    """
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API')
    return groq_api_key

@tool(
    "LowerCaseConverter",
    description="Converts any string to lower case",
)
def str_lower(s:str) -> str:
    """
    This tool is used to convert a string to lowercase
    :param s: the string to convert
    :return: converted string
    """
    return s.lower()

def model():
    """
    This method is used to define the LLM_model
    :return: LLM_model
    """
    llm = ChatGroq(
                model='llama-3.3-70b-versatile',
                api_key=get_api_key(),
                temperature=0
            )
    return llm

def agent_building():
    """
    This method is used to define the agent
    :return: None
    """
    agent = create_agent(
                    model=model(),
                    tools=[str_lower],
                    system_prompt= "You are a helpful assistant that can convert any given string to lower case"
                )
    inputs = {
        "messages":[ HumanMessage(content='ALOK YADAV')],
    }

    response = agent.invoke(inputs)
    print(response)

    final_message = response['messages'][-1]
    print(final_message.content)

def main():
    """
    This is the main function, Entry point
    """
    get_api_key()
    model()
    agent_building()

if __name__ == '__main__':
    main()

