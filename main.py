# Load language model, embeddings, and index for conversational AI
from langchain.chat_models import ChatOpenAI                #model
from langchain.indexes import VectorstoreIndexCreator       #index
from langchain.document_loaders.csv_loader import CSVLoader #tool
from langchain.prompts import PromptTemplate                #prompt
from langchain.memory import ConversationBufferMemory       #memory
from langchain.chains import RetrievalQA                    #chain

import os

def load_llm():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    return llm

def load_index():
    loader = CSVLoader(file_path="uae_holidays.csv")
    index = VectorstoreIndexCreator().from_loaders([loader])
    return index

template = """
You are a assistant to help answer when are the official UAE holidays, 
    based only on the data provided.
Context: {context}
-----------------------
History: {chat_history}
=======================
Human: {question}
Chatbot:
"""

prompt = PromptTemplate(input_variables=["chat_history", "context", "question"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")
qa = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type='stuff',
    retriever=load_index().vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "prompt": prompt,
        "memory": memory
    }
)

def print_response_for_query(query):
    return print(qa.run({"query": query}))

while(True):
    query = input()
    print_response_for_query(query)