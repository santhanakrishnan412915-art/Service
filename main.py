from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
import os
from langgraph.graph import StateGraph, END, START
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import asyncio

app = FastAPI()

origins=["http://localhost:3000","*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origins], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

os.environ["GOOGLE_API_KEY"] = "AIzaSyAvfPE6ggTkfRc1zCtZsGqpSpS_PDwSY2k"

class Query(BaseModel):
    query: str

class State(BaseModel):
    query: str
    result: str
    botresponse: str

def queryentry(state: State):
    query = state.query.lower()
    state.query = query
    return state

async def validate(state: State):
    prompt = ChatPromptTemplate.from_template(
        """classify the intent of the user input{input}
        Possible intents: are greetings , studies,skills,projects,contact,publishes,experience 
        Respond with one of the intents only. By default, respond with 'aboutme' 
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    chain = prompt | llm

    result = await chain.ainvoke({'input': state.query})
    state.result = result.content if hasattr(result, "content") else str(result)
    return state

async def responsestate(state: State):
    query = state.result
    doc = """
        Here are my key skills:
        - Programming Languages: Python, Java, C, SQL
        - Machine Learning: Regression, Classification, Clustering, Random Forest, SVM, PCA
        - Deep Learning: CNNs, RNNs, LSTMs, GANs, Transformers, Attention Models
        - NLP & LLMs: BERT, NER, Text Classification, RAG, LangChain, Prompt Engineering
        - Frameworks & Libraries: TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, SpaCy, NLTK
        - Full-stack: FastAPI, Flask, React.js, AWS, Docker
        - MLOps & Dev Tools: Git, Jupyter, Streamlit, Replit
        Here are some of my key projects:
        1. NeuraSearch – AI-Powered Semantic Search Engine (React.js, FastAPI, AWS, Render)
        2. ATS Bot for Resume Evaluation – Python, NLP, GenAI, Docker, Streamlit
        3. Medicinal Plant Recognition Using CNN – TensorFlow, Flask, Docker
        4. Parallel Attention GAN for Cloud Removal (Satellite Imagery) – TensorFlow, PyTorch
         You can contact me via:
        - Email: krishnashanmugam9159@gmail.com
        - Phone: +91 9345683659
        - LinkedIn: linkedin.com/in/santhanakrishnan-s-b19459275
        - GitHub: github.com/Santhanakrishnan-Shanmugam

         - Book Chapter: 'Generative Adversarial Networks for Remote Sensing' (Scopus Indexed, IGI Global, 2025)

         Deep Learning Research Intern at NIT Karaikal (Jun 2024 – Jul 2024)
        - Optimized GAN-based satellite image enhancement pipelines.
        - Improved training efficiency, reproducibility, and inference performance in research workflows.
    """
    model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", timeout=120)
    docs = [Document(page_content=doc)]
    db = Chroma.from_documents(docs, model)

    retrieved_docs = db.similarity_search(state.query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        """
        You are a friendly and concise AI assistant who directly answers user questions 
        about Santhanakrishnan based on the given context.

        Context:
        {context}

        User Query:
        {query}

        Guidelines:
        - Respond naturally like a helpful support bot.
        - Don’t introduce yourself or say “I understand”.
        - Reply in 1–3 short sentences.
        - If the question isn’t clear, politely ask for clarification.
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    chain = prompt | llm
    result = await chain.ainvoke({'query': query, 'context': context})

    state.botresponse = result.content if hasattr(result, "content") else str(result)
    return state

graph = StateGraph(State)
graph.add_node('get', queryentry)
graph.add_node('validate', validate)
graph.add_node('response', responsestate)

graph.add_edge(START, 'get')
graph.add_edge('get', 'validate')
graph.add_edge('validate', 'response')
graph.add_edge('response', END)

async def work(query: str):
    app = graph.compile()
    final_state = await app.ainvoke({"query": query, "result": "", "botresponse": ""})
    return final_state['botresponse']

@app.post("/")
async def greet(query: Query):
    response = await work(query.query)
    return {"response": response}  
