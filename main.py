
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import requests
import re
from typing import List

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

class RewriteRequest(BaseModel):
    paragraph: str

@app.post("/rewrite")
def rewrite_academic(request: RewriteRequest):
    paragraph = request.paragraph.strip()
    if not paragraph:
        raise HTTPException(status_code=400, detail="Input paragraph is empty.")

    keywords = list(set(re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', paragraph)))
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    rewritten_sentences = []
    structure_used = []

    for sent in sentences:
        if len(sent.strip()) < 10:
            continue

        try:
            query = re.sub(r'[^\w\s]', '', sent)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract,year&limit=5"
            response = requests.get(url)
            response.raise_for_status()
            papers = response.json().get("data", [])
            matched = []
            for paper in papers:
                if paper.get("year") and paper["year"] < 2010:
                    abstract = paper.get("abstract", "")
                    for s in re.split(r'[.!?]', abstract):
                        s_clean = s.strip()
                        if 40 < len(s_clean) < 300:
                            matched.append(s_clean)
            if not matched:
                raise ValueError("No valid structure found.")

            sentence_embedding = model.encode(sent, convert_to_tensor=True)
            matched_embeddings = model.encode(matched, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(sentence_embedding, matched_embeddings)[0]
            best_idx = int(scores.argmax())
            selected_structure = matched[best_idx]
            structure_used.append(selected_structure)
        except Exception as e:
            selected_structure = "The scientific literature addresses a range of environmental concerns, emphasizing the need for coordinated action."
            structure_used.append(selected_structure)

        rewritten = selected_structure
        placeholder_keywords = re.findall(r'\b(?:[a-z]{4,}|[A-Z][a-z]+)\b', rewritten)
        for i, word in enumerate(placeholder_keywords):
            if i < len(keywords):
                rewritten = re.sub(re.escape(word), keywords[i], rewritten, 1)
        rewritten_sentences.append(rewritten)

    return {
        "input": paragraph,
        "rewritten_output": " ".join(rewritten_sentences),
        "structure_matched": structure_used,
        "keywords": keywords
    }

@app.post("/upload_pdf")
def extract_sentences_from_pdf(file: UploadFile = File(...)):
    try:
        doc = fitz.open(stream=file.file.read(), filetype="pdf")
        sentences = []
        for page in doc:
            text = page.get_text()
            if text:
                for sentence in re.split(r'[.!?]', text):
                    if 40 < len(sentence) < 300:
                        sentences.append(sentence.strip())
        return {"sentences": sentences[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")
