import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# Use these modern imports instead of RetrievalQA

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from Rag import get_retriever

load_dotenv()
BASE_URL = os.getenv("BASE_URL")


# def create_qa_chain():
#     """Creates a modern Retrieval Chain."""
#     retriever = get_retriever()
#     llm = OllamaLLM(model="llama3", base_url=BASE_URL)
#
#     # 1. Define how the LLM should use the context
#     system_prompt = (
#         "You are a helpful assistant. Use the following context to answer the question. "
#         "If you don't know the answer, say you don't know. "
#         "\n\n"
#         "{context}"
#     )
#
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])
#
#     # # 2. Create the 'Stuff' chain (it 'stuffs' context into the prompt)
#     # question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     #
#     # # 3. Create the final retrieval chain
#     # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#     rag_chain = ({
#         retriever:RunnablePassthrough()
#     }  |prompt|llm|StrOutputParser())
#     return rag_chain


# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
#
#
# def create_qa_chain():
#     retriever = get_retriever()
#     llm = OllamaLLM(model="mistral-nemo", base_url=BASE_URL)
#
#     # 1. Ensure your prompt has {context} and {question} placeholders
#     template = """Answer the question based only on the following context:
#     {context}
#
#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#
#     # 2. FIXED: Use "context" and "question" as string keys
#     rag_chain = (
#             {
#                 "context": retriever,
#                 "question": RunnablePassthrough()
#             }
#
#             | prompt
#             | llm
#             | StrOutputParser()
#     )
#
#     return rag_chain
#
#
# def ask_question(query: str):
#     """Run a query through the chain and return answer + sources."""
#     # Note: For efficiency, in production you should cache this chain
#     chain = create_qa_chain()
#
#     # Modern chains use 'input' as the key
#     result = chain.invoke({"input": query})
#
#     answer = result["answer"]
#     # Sources are found in the 'context' key
#     sources = list(set([doc.metadata.get("source", "Unknown") for doc in result["context"]]))
#
#     return answer, sources

from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Rag import get_retriever
import os

BASE_URL = os.getenv("BASE_URL")


def create_qa_chain():
    retriever = get_retriever()
    llm = OllamaLLM(model="mistral-nemo", base_url=BASE_URL)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # We use RunnableParallel to keep the context (sources) available in the final output
    rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: x["context"]))

            | prompt
            | llm
            | StrOutputParser()
    )

    rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain


def ask_question(query: str):
    """Run a query through the chain and return answer + sources."""
    chain = create_qa_chain()

    # We pass the string directly to the chain
    result = chain.invoke(query)

    # Now result is a dictionary: {"answer": "...", "context": [...], "question": "..."}
    answer = result["answer"]

    # Extract source metadata from the documents in the context
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in result["context"]]))

    return answer, sources


