from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os, json, re, time
from typing import TypedDict, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

load_dotenv()

try:
    import streamlit as st
    HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))
except:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN missing. Add it in Streamlit secrets or .env.")

LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
hf_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)


def hf_generate(prompt: str) -> str:
    last_err = None
    for attempt in range(4):
        try:
            resp = hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.0
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError("HF inference failed after retries.") from last_err


class ResumeEvalState(TypedDict):
    resume_text: str
    job_description: str
    vectorstore: object
    score: int
    report: str
    improvements: str


def extract_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"```json", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```", "", raw).strip()

    try:
        return json.loads(raw)
    except:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {}


def load_idx_resume(state: ResumeEvalState) -> ResumeEvalState:
    resume_text = state["resume_text"]

    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        model=emb_model,
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN,
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([resume_text])

    vectorstore = FAISS.from_documents(docs, hf_embeddings)
    return {**state, "vectorstore": vectorstore}


def generate_score_and_report(state: ResumeEvalState) -> ResumeEvalState:
    vectorstore = state["vectorstore"]
    job_description = state["job_description"]
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def _try_once() -> Dict[str, Any]:
        retrieved_docs = retriever.invoke(job_description)
        if not retrieved_docs:
            return {"score": 0, "report": "No relevant resume content found — unable to evaluate."}

        context = "\n\n".join([d.page_content for d in retrieved_docs])

        full_prompt = f"""
You are an expert resume evaluator.

Return ONLY valid JSON. No markdown. No explanation.
Return STRICT JSON only. If unsure, still output valid JSON.
Format exactly:
{{ "score": <integer 0-100>, "report": "<3-4 paragraphs>" }}

Job Description:
---
{job_description}
---

Resume Context:
---
{context}
---
"""

        raw_output = hf_generate(full_prompt)
        parsed = extract_json(raw_output)

        if "score" not in parsed or "report" not in parsed:
            raise ValueError("Invalid JSON output")

        score = max(0, min(100, int(parsed["score"])))
        report = str(parsed["report"]).strip()

        if not report:
            raise ValueError("Empty report")

        return {"score": score, "report": report}

    for attempt in range(3):
        try:
            result = _try_once()
            return {**state, "score": result["score"], "report": result["report"]}
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return {**state, "score": 0, "report": "Evaluation failed — model could not produce valid output."}


def generate_suggestions(state: ResumeEvalState) -> ResumeEvalState:
    report = state.get("report", "")
    job_description = state["job_description"]
    score = state.get("score", 0)

    if "Evaluation failed" in report:
        return {**state, "improvements": "Unable to generate suggestions at this time."}

    full_prompt = f"""
You are an expert career coach and resume writer.

Give 5–8 specific actionable suggestions to improve the resume.
Return ONLY bullet points.

Candidate Score: {score}/100

Job Description:
---
{job_description}
---

Evaluation Report:
---
{report}
---
"""

    try:
        suggestions = hf_generate(full_prompt).strip()
        if not suggestions:
            raise ValueError("Empty suggestions")
        return {**state, "improvements": suggestions}
    except Exception:
        return {**state, "improvements": "Unable to generate suggestions at this time."}


def format_final_output(state: ResumeEvalState) -> Dict[str, Any]:
    return {
        "score": state.get("score", 0),
        "report": state.get("report", "No report generated."),
        "improvements": state.get("improvements", "No suggestions available."),
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
