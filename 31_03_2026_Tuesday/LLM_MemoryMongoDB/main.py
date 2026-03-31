import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory

load_dotenv()

MONGO_URI = os.getenv("MONGO_CONN_URL")
DB_Name = "LangChain"
COLLECTION_NAME = "LangChain_context"
SESSION_ID = 'USER'
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Initialize  mongoDB  to store older chat history permanently
mongo_memory = MongoDBChatMessageHistory(
                            connection_string=MONGO_URI,
                            database_name=DB_Name,
                            collection_name=COLLECTION_NAME,
                            session_id=SESSION_ID,
                        )

class ShortTermMemory:
    """
    Manages recent messages in RAM and moves old ones to MongoDB when full.
    """
    def __init__(self,max_size):
        self.memory = deque(maxlen=max_size)

    def add_message(self, message):
        """
        Adds a new message to short-term memory (RAM).
        If RAM is full, the oldest message is moved to MongoDB.
        """
        if len(self.memory) == self.memory.maxlen:
            oldest_msg = self.memory.popleft()
            mongo_memory.add_message(oldest_msg)

        self.memory.append(message)

    def get_messages(self):
        """
        Returns the current messages stored in RAM as a list
        :return:
        """
        return list(self.memory)

# loading context from MongoDB
def get_long_term_context(limit=4):
    """
    Retrieves the last few messages store in MongoDB.
    """
    messages = mongo_memory.messages
    if not messages:
        return []
    return messages[-limit:]

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3
)

memory = ShortTermMemory(max_size=4)

def chat():
    """
    Main loop to handle user input and AI responses.
    """
    print("\n #### WELCOME TO CHAT (type 'exit' to stop) #### \n")

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Gather history from both RAM and Database
        short_context = memory.get_messages()
        long_context = get_long_term_context()

        # Build the final prompt structure
        system_message = [SystemMessage(content="You are helpful assistant.")]
        human_message = HumanMessage(content=user_input)

        # Combine: System + Old History + Recent History + Current Question
        final_context = system_message + long_context + short_context + [human_message]

        response = llm.invoke(final_context)
        print(f"\n AI: {response.content} \n")

        # Save the exchange to memory
        memory.add_message(human_message)
        memory.add_message(AIMessage(content=response.content))

if __name__ == "__main__":
    chat()