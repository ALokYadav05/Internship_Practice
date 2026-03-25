from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('sample_data')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
                                        # to split data by 50 chunks, but it will not split in middle of any words
                                        chunk_size=50,
                                        chunk_overlap=20
                                    )
splits = text_splitter.split_documents(documents)
print(documents)  #raw-data
print(splits)      #data after its split

for i,chunk in enumerate(splits):
    print(f"Chunk {i+1} : ", end=" ")
    print(chunk.page_content)

context_text = documents[0].page_content   # taking raw-data as a context to pass in PromptTemplate

# #prompt-template
# template =  """
#             Answer the question strictly based on the provided context.
#             If you didn't find context in the provided data, simply say "I don't know"
#
#             Context: {context}
#             Question: {question}
#             """

# prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer strictly based on the context."
                   "If the answer isn't there, say 'I don't know'."),
        ("human", "Context: {context} \n Question: {question}")
    ])

print(f"Prompt: {prompt}")

llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API"),
        temperature=0.6,
        max_tokens=100
    )

chain = prompt | llm     # chaining

question="how statistics is used used in data science?"
result = chain.invoke({
            "context": context_text ,
            "question": question,
        })

print(f" \n Result: {result.content}")
print(f" \n Prompt-token: {result.response_metadata['token_usage']['prompt_tokens']}")
print(f" \n Output-token: {result.usage_metadata['output_tokens']}")
print(f" \n Total-token: {result.usage_metadata['total_tokens']}")