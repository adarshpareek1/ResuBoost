# 📄 ResuBoost

An intelligent resume analysis tool that evaluates how well your resume matches a job description using AI and provides actionable improvement suggestions.

## 🌟 Features

- **Smart Matching**: AI-powered analysis comparing resume content against job requirements
- **Score Generation**: Get a match score from 0-100
- **Detailed Reports**: Comprehensive evaluation highlighting strengths and gaps
- **Actionable Suggestions**: 5-8 specific recommendations to improve your resume
- **PDF Support**: Upload resumes in PDF format
- **User-Friendly Interface**: Clean, intuitive Streamlit web interface

## 🚀 Live Demo

https://resuboost-etskzj6vub6mdgijs9mznu.streamlit.app/

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Framework**: LangChain, LangGraph
- **LLM**: Mistral Mixtral-8x7B via HuggingFace
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **PDF Processing**: PyPDF

## 📋 Prerequisites

- Python 3.9+
- HuggingFace API Token ([Get one here](https://huggingface.co/settings/tokens))

## 📖 How to Use

1. **Upload Resume**: Click "Upload Resume" and select your PDF file
2. **Paste Job Description**: Copy and paste the complete job description
3. **Evaluate**: Click "Evaluate Resume" button
4. **Review Results**: Get your match score, detailed report, and improvement suggestions
5. **Improve**: Use the suggestions to tailor your resume

## 📁 Project Structure

```
resume-evaluator/
├── backend.py           # Core evaluation logic using LangGraph
├── frontend.py          # Streamlit UI
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (local only)
└── README.md           # Project documentation
```

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for accessing models |

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [HuggingFace](https://huggingface.co/) models
- UI created with [Streamlit](https://streamlit.io/)

---

⭐ Star this repo if you find it helpful!
