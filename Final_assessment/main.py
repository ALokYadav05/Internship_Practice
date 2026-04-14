import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_mistralai import ChatMistralAI
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from mongoDB_storage import memory

load_dotenv()

MISTRAL_NEMO = os.getenv("MISTRAL_NEMO")
OLLAMA_MODEL = os.getenv("OLLAMA_LATEST")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
BASE_URL=os.getenv("BASE_URL")

def main ():
    """
    This is the main function
    """

    try:
        file_path = ["./Data/company_policies.pdf","./Data/faq.pdf","./Data/product_manual.pdf"]
        all_data=[]

        for path in file_path:
            loader = PyPDFLoader(path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            split_text = splitter.split_documents(docs)

            all_data.extend(split_text)

        embedder = OllamaEmbeddings(model=EMBEDDING_MODEL,base_url=BASE_URL)

        vector_store = Chroma.from_documents(
                                documents=all_data,
                                embedding=embedder,
                                persist_directory="./chroma")

        retriever = vector_store.as_retriever(
                                search_type='mmr',
                                search_kwargs= {
                                    'k' : 2,
                                    'fetch_k' : 10,
                                    'lambda_mult' : 0.75
                                }
                            )

        system_prompt = """
                        ## You are a helpful RAG Chatbot.
                        ## Rules:
                           Respond normally to greetings like (hi,hello).
                           If query is knowledge based, use the context to answer the question.
                           If user ask about previous conversation, summarize it and give it to user.
                        """

        llm = ChatOllama(
            model=MISTRAL_NEMO,
            temperature=0.25,
            base_url=BASE_URL,
        )

        agent = create_agent(
                        model=llm,
                        system_prompt=system_prompt,
                        middleware=[
                            SummarizationMiddleware(
                                model=llm,
                                trigger=[('messages',6)],
                                keep=('messages',6)
                            )
                        ],
            checkpointer=memory

                    )

        session_id = "user_1"
        config = {'configurable': {'thread_id':session_id}}

        print("\n\n #### --- Welcome to CHATBOT --- #### \n\n")
        print("Enter your Query (Type exit to close):")

        try:
            while True:
                query = input("You: \n")

                if query.lower() == "exit":
                    break

                docs = retriever.invoke(query)
                context = "".join([doc.page_content for doc in docs])

                user_input = f"""
                             Answer the Question from the following context below:
                             context
                             {context}
                             
                             query
                             {query}
                             """

                messages = [{'role':'user','content': user_input}]


                result = agent.invoke({
                                "messages": messages
                            }, config=config)

                print(f"Result : {result['messages'][-1].content}")

        except Exception as e:
            print(f"Error in while-loop: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
