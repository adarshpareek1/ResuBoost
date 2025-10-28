from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os, json, re, time
from typing import TypedDict, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except:
    HF_TOKEN = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=True,
    provider="auto",
)

model = ChatHuggingFace(llm=llm)

class ResumeEvalState(TypedDict):
    resume_text: str
    job_description: str
    vectorstore: object
    score: int
    report: str
    improvements: str

def load_idx_resume(state: ResumeEvalState) -> ResumeEvalState:
    resume_text = state['resume_text']

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([resume_text])
    vectorstore = FAISS.from_documents(docs, hf_embeddings)

    return {**state, "vectorstore": vectorstore}

def generate_score_and_report(state: ResumeEvalState) -> ResumeEvalState:
    vectorstore = state["vectorstore"]
    job_description = state["job_description"]
    retriever = vectorstore.as_retriever(k=5)

    system_prompt = (
        "You are an expert resume evaluator. Compare the candidate's resume excerpts "
        "to the job description and output your answer ONLY as a JSON object:\n\n"
        "{{\n"
        '  "score": <integer from 0 to 100>,\n'
        '  "report": "<3–4 paragraphs explaining key strengths and gaps>"\n'
        "}}\n\n"
        "Be concise, factual, and base your evaluation strictly on the provided resume context."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Job Description:\n---\n{job_description}\n---\n\n"
            "Evaluate the candidate's resume excerpts below for fit:\n"
            "{context}"
        )),
    ])

    def _try_once() -> Dict[str, Any]:
        retrieved_docs = retriever.invoke(job_description)
        if not retrieved_docs:
            return {"score": 0, "report": "No relevant resume content found — unable to evaluate."}

        context = "\n\n".join([d.page_content for d in retrieved_docs])

        rag_chain = (prompt | model.bind(stop=["<|eot_id|>", "<|end_of_text|>"]) | StrOutputParser())
        raw_output = rag_chain.invoke({"job_description": job_description, "context": context})

        try:
            result = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*"score"[^{}]*"report"[^{}]*\}', raw_output, re.DOTALL)
            if not match:
                match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            
            if match:
                try:
                    result = json.loads(match.group(0))
                except Exception:
                    result = {}
            else:
                score_match = re.search(r'"score"\s*:\s*(\d+)', raw_output)
                report_match = re.search(r'"report"\s*:\s*"([^"]+)"', raw_output, re.DOTALL)
                
                if score_match and report_match:
                    result = {
                        "score": int(score_match.group(1)),
                        "report": report_match.group(1)
                    }
                else:
                    result = {}

        if isinstance(result, dict) and "score" in result and "report" in result:
            return result
        else:
            raise ValueError(f"Invalid or missing keys. Got: {list(result.keys())}")

    for attempt in range(3):
        try:
            result = _try_once()
            if result["score"] >= 0:  
                return {**state, "score": result["score"], "report": result["report"]}
        except Exception:
            if attempt < 2:  
                time.sleep(2 ** attempt)

    return {**state, "score": 0, "report": "Evaluation failed — model could not produce valid output."}


def generate_suggestions(state: ResumeEvalState) -> ResumeEvalState:
    report = state["report"]
    job_description = state["job_description"]
    score = state["score"]

    system_prompt = (
        "You are an expert career coach and resume writer. Based on the evaluation report "
        "and the job description, provide 5–8 SPECIFIC, actionable suggestions to improve "
        "the candidate's resume. Output in a clear bullet list format."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", (
            "Candidate Score: {score}/100\n\n"
            "Job Description:\n---\n{job_description}\n---\n\n"
            "Evaluation Report:\n---\n{report}\n---\n\n"
            "Now list improvements:"
        )),
    ])

    suggestion_chain = (prompt | model.bind(stop=["<|eot_id|>", "<|end_of_text|>"]) | StrOutputParser())
    
    try:
        suggestions_text = suggestion_chain.invoke({
            "score": score,
            "job_description": job_description,
            "report": report
        })
        return {**state, "improvements": suggestions_text}
    except Exception:
        return {**state, "improvements": "Unable to generate suggestions at this time."}


def format_final_output(state: ResumeEvalState) -> Dict[str, Any]:
    return {
        "score": state.get("score", 0),
        "report": state.get("report", "No report generated."),
        "improvements": state.get("improvements", "No suggestions available.")
    }

workflow = StateGraph(ResumeEvalState)

workflow.add_node("setup", load_idx_resume)
workflow.add_node("scoring", generate_score_and_report)
workflow.add_node("suggestions", generate_suggestions)
workflow.add_node("formatter", format_final_output)

workflow.add_edge(START, "setup")
workflow.add_edge("setup", "scoring")
workflow.add_edge("scoring", "suggestions")
workflow.add_edge("suggestions", "formatter")
workflow.add_edge("formatter", END)

res_eval_app = workflow.compile()