
import os
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
import streamlit as st
from dotenv import load_dotenv
import json
from datetime import datetime
load_dotenv()

from datetime import datetime, timezone

class AzureOpenAIClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_KEY_1"),
            api_version="2024-06-01",
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
        )

    def generate_streaming_response(self, prompt, model="gpt-4o", system_prompt=""):
        response_text = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            for chunk in self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=4096
            ):
                if len(chunk.choices) > 0:
                    if chunk.choices[0].delta.content is not None:
                        response_text += chunk.choices[0].delta.content
                        message_placeholder.markdown(response_text + "▌")
        return response_text

def get_current_time():
    return datetime.now(timezone.utc).isoformat()

def create_conversational_prompt(history, conversation_length=10):
    conversational_prompt="" 
    for segment in history[-conversation_length:]:
        if segment["RAG_context"] != "":
            conversational_prompt+=f'''
{segment["role"]}:
{segment["RAG_context"]}
''' 
        else:
            conversational_prompt+=f'''
{segment["role"]}:
{segment["content"]}
''' 
    return conversational_prompt

def get_elasticsearch_results(es_client, query, index, size):

    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "nested": {
                        "path": "processed_article_content.inference.chunks",
                        "query": {
                            "sparse_vector": {
                                "inference_id": "elser_v2",
                                "field": "processed_article_content.inference.chunks.embeddings",
                                "query": query
                            }
                        },
                        "inner_hits": {
                            "size": 2,
                            "name": "fsi_cna_business_processed.processed_article_content",
                            "_source": [
                                "processed_article_content.inference.chunks.text"
                            ]
                        }
                    }
                }
            }
        },
        "size": 10
    }

    result = es_client.search(index=index, body=es_query)
    return result["hits"]["hits"]

def create_RAG_context(results, query):
    context = ""
    for hit in results:
        index = hit['_index']
        filename = hit['_source'].get('filename', 'Unknown')
        context += f"\nContext Filename: {filename}\n"
        
        inner_hit_path = f"{index}.body"

        context_arr=[]

        if 'inner_hits' in hit and inner_hit_path in hit['inner_hits']:
            context_arr.append('\n --- \n'.join(inner_hit['_source']['text'] for inner_hit in hit['inner_hits'][inner_hit_path]['hits']['hits']))
        else:
            context_arr.append(json.dumps(hit['_source'], indent=2))
        
        context="".join(context_arr)+"\n"

    prompt = f"""
    Instructions:
    
    - You are an assistant for question-answering tasks.
    - Answer questions truthfully and factually using only the context presented.
    - If you don't know the answer, just say that you don't know, don't make up an answer.
    - Use markdown format for code examples.
    - You are correct, factual, precise, and reliable.
    
    Context:
    {context}

    Query:
    {query}
    
    """
    return prompt

try:
    # Elasticsearch setup
    es_endpoint = os.environ.get("ELASTIC_ENDPOINT")
    es_client = Elasticsearch(
        es_endpoint,
        api_key=os.environ.get("ELASTIC_API_KEY")
    )
except Exception as e:
    es_client=None

LLM = AzureOpenAIClient()

# Page config
st.set_page_config(layout="wide", page_title="Streamlit Chat App")

index="fsi_cna_business_processed"
es_size=5
conversation_length=5
# CHAT WINDOW 
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"{message['content']}")
        st.caption(f"Sent at {message['time']}")

if prompt := st.chat_input("Start Chatting!"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"Sent at {get_current_time()}")

    with st.spinner("Generating Response..."):
        
        elasticsearch_results = get_elasticsearch_results(es_client, prompt, index, es_size)
        RAG_context = create_RAG_context(elasticsearch_results, prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "RAG_context": RAG_context, "time": get_current_time()})

        conversation_prompt = create_conversational_prompt(st.session_state.messages, conversation_length=conversation_length)

        assistant_response = LLM.generate_streaming_response(conversation_prompt)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "RAG_context": "", "time": get_current_time()})
    st.rerun()