import os
from elasticsearch import Elasticsearch
import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timezone
load_dotenv()

try:
    # Elasticsearch setup
    es_endpoint = os.environ.get("ELASTIC_ENDPOINT")
    es_client = Elasticsearch(
        es_endpoint,
        api_key=os.environ.get("ELASTIC_API_KEY")
    )
except Exception as e:
    es_client=None

def es_chat_completion(prompt):
    response = es_client.inference.inference(
        inference_id = os.environ.get("INFERENCE_ID"),
        task_type = "completion",
        input = prompt,
        timeout="180s"
    )
    return response['completion'][0]['result']

def get_current_time():
    return datetime.now(timezone.utc).isoformat()

def get_elasticsearch_results(es_client, query):

    es_query = # FILL IN THE QUERY - Use the Query from Kibana Playground :D #

    result = es_client.search(body=es_query)
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

st.set_page_config(layout="wide")

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
        
        elasticsearch_results = get_elasticsearch_results(es_client, prompt)
        RAG_context = create_RAG_context(elasticsearch_results, prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "RAG_context": RAG_context, "time": get_current_time()})
        assistant_response = es_chat_completion(prompt)

    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "RAG_context": "", "time": get_current_time()})
    st.rerun()