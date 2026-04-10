import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
url = os.getenv("MONGO_CONN_URL")

client = MongoClient(url)
db = client['rag_chatbot']
history = db["history"]

def save_message(session_id, role, content):
    """
    This function saves a message to the database
    :param session_id: session ID
    :param role: role like (user or AI)
    :param content: actual data
    :return: None
    """
    try:
        history.insert_one(
                       {"session_id": session_id,
                        "role": role,
                        "content": content})
    except Exception as e:
        print(f"Error while saving message: {e}")

def get_all_messages(session_id):
    """
    This function gets all messages from the database
    :param session_id: session ID
    :return: None
    """
    try:
        history.find_one({"session_id": session_id})
    except Exception as e:
        print(f"Error while getting messages: {e}")
