import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from backend import res_eval_app

st.set_page_config(page_title="ResuBoost",page_icon="📄",layout="wide")

if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'evaluation_done' not in st.session_state:
    st.session_state.evaluation_done = False
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None

st.title("📄ResuBoost : Your trusted AI Resume Evaluator")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF format)", 
        type="pdf",
        help="Upload a PDF version of your resume"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner("📖 Reading your resume..."):
                loader = PyPDFLoader(tmp_file_path, mode='single')
                documents = loader.load()
                
                if documents:
                    st.session_state.resume_text = documents[0].page_content
                    st.success(f"✅ Resume loaded successfully! ({len(documents[0].page_content)} characters)")
                    
                    with st.expander("👁️ Preview Resume Content"):
                        st.text_area(
                            "Resume Text", 
                            st.session_state.resume_text[:1000] + "..." if len(st.session_state.resume_text) > 1000 else st.session_state.resume_text,
                            height=200,
                            disabled=True
                        )
                else:
                    st.error("❌ No content found in the PDF")
                    
        except Exception as e:
            st.error(f"❌ An error occurred during PDF loading: {e}")
            
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

with col2:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="Paste the full job description including responsibilities, requirements, and qualifications...",
        help="Provide a detailed job description for accurate evaluation"
    )
    
    if job_description:
        st.info(f"📝 Job description length: {len(job_description)} characters")

st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    evaluate_btn = st.button(
        "🚀 Evaluate Resume",
        type="primary",
        use_container_width=True,
        disabled=not (st.session_state.resume_text and job_description)
    )

if evaluate_btn:
    if not st.session_state.resume_text:
        st.error("⚠️ Please upload a resume first!")
    elif not job_description:
        st.error("⚠️ Please provide a job description!")
    else:
        with st.spinner("🔄 Analyzing your resume... This may take 30-60 seconds"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Building knowledge base...")
                progress_bar.progress(25)
                
                initial_state = {
                    "resume_text": st.session_state.resume_text,
                    "job_description": job_description
                }
                
                status_text.text("Generating evaluation report...")
                progress_bar.progress(50)
                
                result = res_eval_app.invoke(initial_state)
                
                status_text.text("Creating improvement suggestions...")
                progress_bar.progress(75)
                
                st.session_state.evaluation_result = result
                st.session_state.evaluation_done = True
                
                progress_bar.progress(100)
                status_text.text("✅ Evaluation complete!")
                
            except Exception as e:
                st.error(f"❌ An error occurred during evaluation: {e}")
                st.exception(e)

if st.session_state.evaluation_done and st.session_state.evaluation_result:
    st.markdown("---")
    st.header("📊 Evaluation Results")
    
    result = st.session_state.evaluation_result
    score = result.get("score", 0)
    report = result.get("report", "No report available.")
    improvements = result.get("improvements", "No suggestions available.")
    
    if score >= 70:
        st.success(f"### 🎉 Excellent Match! Score: {score}/100")
    elif score >= 50:
        st.warning(f"### 👍 Good Match. Score: {score}/100")
    else:
        st.error(f"### ⚠️ Needs Improvement. Score: {score}/100")
    
    st.markdown("---")
    
    col_report, col_suggestions = st.columns([1, 1])
    
    with col_report:
        st.subheader("📝 Detailed Report")
        st.info(report)
    
    with col_suggestions:
        st.subheader("💡 Improvement Suggestions")
        st.info(improvements)
    
    st.markdown("---")
    
    if st.button("🔄 Evaluate Another Resume"):
        st.session_state.evaluation_done = False
        st.session_state.evaluation_result = None
        st.session_state.resume_text = None
        st.rerun()
        
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI-powered tool evaluates how well your resume matches a job description.
    
    **How it works:**
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click 'Evaluate Resume'
    4. Get a detailed match score and improvement suggestions
    """)
    
    st.markdown("---")
    st.subheader("📌 Tips")
    st.write("""
    - Ensure your resume is clear and readable
    - Include the complete job description
    - Use the suggestions to tailor your resume
    - Re-evaluate after making changes
    """)
    
    st.markdown("---")
    st.caption("Powered by LangGraph & HuggingFace")