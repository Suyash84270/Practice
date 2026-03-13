from typing import TypedDict, List

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from rag_pipeline import build_rag_pipeline
from src.logger.logger import get_logger

logger = get_logger(__name__)


# ---------------------------
# Prompt Template
# ---------------------------

PROMPT = PromptTemplate(
    template="""
You are an AI assistant specialized in construction documents.

Answer the user's question using ONLY the provided context.

If the answer is not in the document say:
"Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# ---------------------------
# LLM Loader
# ---------------------------

def load_llm():
    """
    Load LLM for RAG QA.

    Default: Local HuggingFace model (TinyLlama)

    You can switch to OpenAI by uncommenting the OpenAI section.
    """

    # logger.info("Loading local HuggingFace LLM")

    # generator = pipeline(
    #     "text-generation",
    #     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     max_new_tokens=256,
    #     temperature=0.2
    # )

    # return HuggingFacePipeline(pipeline=generator)


    # ---------------------------------------------------
    # OPTIONAL: Use OpenAI instead of local model
    # ---------------------------------------------------
    #
    # Requirements:
    # pip install langchain-openai
    #
    # Set environment variable:
    # export OPENAI_API_KEY="your_api_key"
    #
    from langchain_openai import ChatOpenAI
    #
    logger.info("Loading OpenAI LLM")
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )


# ---------------------------
# Graph State
# ---------------------------

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    documents: List


# ---------------------------
# Chatbot Builder
# ---------------------------

def create_chatbot(file_path: str):
    """
    Build RAG chatbot using LangGraph.

    Steps:
    1. Build vector database from documents
    2. Retrieve relevant chunks
    3. Generate answer using LLM
    """

    logger.info("Building RAG pipeline")

    vector_store = build_rag_pipeline(file_path)

    retriever = vector_store.as_retriever()

    llm = load_llm()


    # -------- Retrieve Node --------

    def retrieve(state: GraphState):

        docs = retriever.invoke(state["question"])

        context = "\n\n".join([d.page_content for d in docs])

        return {
            "context": context,
            "documents": docs
        }


    # -------- Generate Node --------

    def generate(state: GraphState):

        prompt = PROMPT.format(
            context=state["context"],
            question=state["question"]
        )

        answer = llm.invoke(prompt)

        return {"answer": answer}


    # -------- LangGraph Workflow --------

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    logger.info("Chatbot ready")

    return workflow.compile()