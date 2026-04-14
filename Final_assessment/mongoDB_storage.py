import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()
url = os.getenv("MONGO_CONN_URL")

client = MongoClient(url)
db = client['rag_chatbot']
collection = db["history"]

memory = MongoDBSaver(client=client,collection_name = collection)     # creating mongodb object