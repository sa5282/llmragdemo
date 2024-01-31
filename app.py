import os
import requests
import chromadb
import chainlit as cl

API_TOKEN = os.environ['API_TOKEN'] #Set a API_TOKEN environment variable before running
print(f"API_TOKEN -> {API_TOKEN}")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha" #Add a URL for a model of your choosing
headers = {"Authorization": f"Bearer {API_TOKEN}"}

@cl.on_chat_start
def on_chat_start():
    #cl.user_session.set("counter", 0)
    print("Intializing ChromaDB...")
    initialize_chromadb()
    print("ChromaDB initialization complete.")
    print("Loading docs into ChromsDB...")
    add_docs_to_collection()
    print("Documents loaded into ChromaDB successfully.")

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    context_doc = get_relevant_document_from_db(message.content)
    prompt = construct_prompt(message.content, context_doc)
    response = query(prompt)


    # Send a response back to the user
    await cl.Message(
        content=f"Received: {response}",
    ).send()


def query(prompt):
    payload = {
        "inputs": prompt,
        "parameters": { #Try and experiment with the parameters
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": False,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']

def initialize_chromadb():

    try:
        print("Intializing the ChromaDB Client...")
        chroma_client = chromadb.Client()
        chroma_client.delete_collection("documents")
    except ValueError:
        """documents collection doesn't exist, create the collection next"""
    finally:
        chroma_client.create_collection(name="documents")

def add_docs_to_collection():
    add_file("resources/baseball.md", {"type": "board game"}, "baseball")
    add_file("resources/house.md", {"type": "structure"}, "house")


def add_file(relative_path, metadata, id):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_collection("documents")
    with open(relative_path) as data_file:
        data = data_file.read()
    collection.add(
        documents=[data],
        metadatas=[metadata],
        ids=[id]
    )

def get_relevant_document_from_db(question):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_collection("documents")
    result = collection.query(query_texts=[question], n_results=1)
    doc_retrieved = result["documents"]
    id_retrieved = result["ids"]
    print(f"Document ID returned by ChromaDB: {id_retrieved}")
    return doc_retrieved[0][0]


def construct_prompt(question, context):
    prompt = f"""Use the following context to answer the question at the end.

    {context}

    Question: {question}
    """
    return prompt

#print(query(prompt))

def driver():
    question = input("Type your question here:")
    while question != "quit":
        context_doc = get_relevant_document_from_db(question)
        prompt = construct_prompt(question, context_doc)
        #print(prompt)
        print(query(prompt))
        question = input("Type another question here: ")

#if __name__ == "__main__":
    #print("Main function called....")
    #initialize_chromadb()
    #add_docs_to_collection()
    #driver()
