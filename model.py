import fitz
import re
from main import PDFOutline
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from openai import OpenAI

class Model:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def assign_span_sections(self, tp):
        sections = []
        lines_by_page, outline_json = PDFOutline(self.pdf_path).analyse()
        outline = json.loads(outline_json)
        total_pages = tp - 1
        items = outline.get("outline", [])
        n = len(items)

        # index lines on each page
        for p in lines_by_page:
            for i, d in enumerate(lines_by_page[p]):
                d["index"] = i

        def find_start_index(page, heading_text):
            for line in lines_by_page.get(page, []):
                if line.get("text", "").strip().lower() == heading_text.strip().lower():
                    return line["index"]
            return 0

        # ensure every outline item has start_index
        for item in items:
            item["start_index"] = find_start_index(item.get("page"), item.get("text", ""))

        # compute end_index and end_page robustly
        for i, item in enumerate(items):
            start_page = item.get("page")
            start_index = item.get("start_index", 0)

            if i + 1 < n:
                next_item = items[i + 1]
                next_page = next_item.get("page")
                next_start = next_item.get("start_index", None)

                if next_page == start_page:
                    if next_start is None:
                        # fallback to last line on page
                        next_start = len(lines_by_page.get(start_page, []))
                    end_index = max(0, next_start - 1)
                    end_page = start_page
                else:
                    # next heading on later page
                    if next_start is None:
                        end_index = len(lines_by_page.get(next_page, [])) - 1 if lines_by_page.get(next_page) else 0
                    else:
                        end_index = max(0, next_start - 1)
                    end_page = next_page
            else:
                # last heading: until end of document
                end_page = total_pages
                end_index = len(lines_by_page.get(end_page, [])) - 1 if lines_by_page.get(end_page) else 0

            item["start_index"] = start_index
            item["end_index"] = end_index
            item["end_page"] = end_page

        # build sections
        for h in items:
            s = {
                "heading": h.get("text", ""),
                "start_page": h.get("page"),
                "end_page": h.get("end_page"),
                "start_index": h.get("start_index", 0),
                "end_index": h.get("end_index", 0)
            }

            start = s["start_page"]
            end = s["end_page"]

            if start == end:
                lines = ""
                for line in lines_by_page.get(start, []):
                    if line["index"] < s["start_index"]:
                        continue
                    if line["index"] > s["end_index"]:
                        break
                    lines += " " + line.get("text", "")
                s["content"] = lines
            else:
                parts = []
                for p in range(start, end + 1):
                    lines = ""
                    for line in lines_by_page.get(p, []):
                        if p == start and line["index"] < s["start_index"]:
                            continue
                        if p == end and line["index"] > s["end_index"]:
                            break
                        lines += " " + line.get("text", "")
                    parts.append(lines)
                s["content"] = "\n".join(parts)

            sections.append(s)

        return sections
    
    # in case heasing detection for the document fails
    def fallback_page_sections(self):
        sections = []
        lines_by_page, outline_json = PDFOutline(self.pdf_path).analyse()
        for p, lines in lines_by_page.items():
            text = " ".join(l["text"] for l in lines).strip()
            if not text:
                continue
            sections.append({
                "heading": f"Page {p+1}",
                "start_page": p,
                "end_page": p,
                "content": text
            })
        return sections
    
    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.split()) >= 5]
    
    def semantic_chunk_sections(self, sections, min_words=60, max_words=120, overlap_sentences=1):
        chunks = []

        for i, s in enumerate(sections):
            sentences = self.split_into_sentences(s["content"])
            if not sentences:
                continue

            start = 0
            chunk_id = 0

            while start < len(sentences):
                words = []
                end = start

                # pack sentences until max_words
                while end < len(sentences):
                    words.extend(sentences[end].split())
                    if len(words) >= max_words:
                        break
                    end += 1

                # enforce minimum length
                if len(words) < min_words:
                    break

                chunk_text = " ".join(sentences[start:end + 1])

                chunks.append({
                    "chunk_id": f"s{i}_c{chunk_id}",
                    "section_heading": s["heading"],
                    "page_start": s["start_page"],
                    "page_end": s["end_page"],
                    "text": chunk_text
                })

                chunk_id += 1
                start = max(start + 1, end - overlap_sentences)

        return chunks
    def overlapping_chunk_sections(self, sections, chunk_size=300, overlap_ratio=0.3):
        chunks = []
        overlap = int(chunk_size * overlap_ratio)
        for i, s in enumerate(sections):
            words = s["content"].split()
            total = len(words)
            if total == 0:
                continue
            start = 0
            id = 0
            while start < total:
                end = min(start + chunk_size, total)
                chunk_text = " ".join(words[start: end]).strip()
                if not chunk_text:
                    break
                chunks.append({
                    "chunk_id": f"s{i}_c{id}",
                    "section_heading": s["heading"],
                    "page_start": s["start_page"],
                    "page_end": s["end_page"],
                    "text": chunk_text
                })

                id += 1
                if end == total:
                    break
                start = end - overlap
        return chunks

    def build_bm25_index(self, chunks):
        def normalize(text):
            text = text.lower()
            text = re.sub(r"[^\w\s]", " ", text)
            return text.split()

        tokenized_corpus = [
            normalize(c["text"])
            for c in chunks
        ]

        return BM25Okapi(tokenized_corpus)
    
    def bm25_retrieve(self, bm25, chunks, query, top_k=200):
        query = re.sub(r"[^\w\s]", " ", query.lower())
        query_tokens = query.split()

        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**chunks[i], "bm25_score": float(scores[i])}
            for i in top_indices
        ]
    
    def load_embedding_model(self, model_name="all-mpnet-base-v2"):
        self.embed_model = SentenceTransformer(model_name)
    
    def build_embedding_index(self, chunks, batch_size=32):
        texts = [
            f"Section: {c['section_heading']}. Content: {c['text']}"
            for c in chunks
        ]

        embeddings = self.embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings
    
    def embedding_retrieve(self, embeddings, chunks, query, top_k=200):
        query_emb = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **chunks[idx],
                "semantic_score": float(scores[idx])
            })

        return results
    
    def load_cross_encoder(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.ce_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cpu")
        self.ce_model.eval()
    
    def cross_encoder_rank(self, candidates, query, persona, job, top_k=20, batch_size=8):
        scores = []

        with torch.no_grad():
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]

                pairs = [
                    (
                        f"Persona: {persona}. Task: {job}. Query: {query}",
                        f"Section title: {c['section_heading']}. Content: {c['text']}"
                    )
                    for c in batch
                ]

                encoded = self.ce_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to("cpu")

                outputs = self.ce_model(**encoded)

                # Use CLS token
                logits = outputs.logits.squeeze(-1)

                for c, s in zip(batch, logits):
                    scores.append({
                        **c,
                        "cross_score": float(s),
                        "semantic_score": float(c["semantic_score"]),
                        "bm25_score": float(c["bm25_score"])
                    })

        reranked = sorted(
            scores,
            key=lambda x: x["cross_score"],
            reverse=True
        )

        return reranked[:top_k]
    
    def aggregate_chunk_into_sections(self, ranked_chunks):
        map = defaultdict(list)
        for c in ranked_chunks:
            key = (c["section_heading"], c["page_start"], c["page_end"])
            map[key].append(c)
        
        sections = []
        for (heading, ps, pe), chunks in map.items():
            imp = max(c["cross_score"] for c in chunks)
            cross = sum(c["cross_score"] for c in chunks) / len(chunks)
            semantic = sum(c["semantic_score"] for c in chunks) / len(chunks)
            bm25 = sum(c["bm25_score"] for c in chunks) / len(chunks)
            sections.append(
                {
                    "section_heading": heading,
                    "page_start": ps,
                    "page_end": pe,
                    "importance_score": imp,
                    "cross_score": cross,
                    "semantic_score": semantic,
                    "bm25_score": bm25,
                    "chunks": chunks
                }
            )
        sections.sort(key=lambda x: x["importance_score"], reverse=True)
        for i, s in enumerate(sections, start=1):
            s["importance_rank"] = i
        return sections
    



def merge(bm25, semantic):
    merged = {}
    for r in bm25:
        merged[r["chunk_id"]] = r
    for r in semantic:
        if r["chunk_id"] in merged:
            merged[r["chunk_id"]]["semantic_score"] = r["semantic_score"]
        else:
            merged[r["chunk_id"]] = r
    for r in merged.values():
        r.setdefault("semantic_score", 0.0)
        r.setdefault("bm25_score", 0.0)
    return list(merged.values())


collections = [str(p) for p in Path("./test/files/collection2").glob("*.pdf")]

p = Path("./test/files/collection2/json/challenge1b_input (1).json")
with p.open("r", encoding="utf-8") as f:
    input = json.load(f)

persona = input["persona"]["role"]
task = input["job_to_be_done"]["task"]
files = str([i["title"] for i in input["documents"]])


def generate_queries_with_llm():
    openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    model="llama3.2",
    response_format={
            "type": "json_object"
    },
    messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert information retrieval assistant. "
                    "You specialize in converting user intent into effective search queries "
                    "for keyword search, semantic search, and reranking systems."
                    "Return ONLY valid JSON with keys: bm25_query, semantic_query, cross_query. NOTHING ELSE MUST BE RETURNED"
                )
            },
            {
                "role": "user",
                "content": (
                    "Given the following inputs:\n\n"
                    f"Persona: {persona}\n"
                    f"Job to be done: {task}\n"
                    f"Document headings: {files}\n(These are the headings of the documents which the ranking pipeline will process)\n"
                    "Generate THREE queries in JSON format:\n"
                    "1. bm25_query: short keyword-based query (nouns only, no stopwords)\n"
                    "2. semantic_query: natural language query capturing full intent\n"
                    "3. cross_query: concise search-style query suitable for reranking\n\n"
                    "Return ONLY valid JSON with keys: bm25_query, semantic_query, cross_query. NOTHING ELSE MUST BE RETURNED"
                )
            }
        ]
    try:
        chat_completion = openai.chat.completions.create(
            messages=messages,
            model='llama3.2', 
            stream=False,
        )
        response = chat_completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        return e
    
response = generate_queries_with_llm()

raw = response.strip()
raw = raw.replace("```json", "").replace("```", "").strip()
response = json.loads(raw)
bm25_query = response["bm25_query"]
if isinstance(bm25_query, list):
    bm25_query = " ".join(bm25_query)
semantic_query = str(response["semantic_query"])
cross_query = str(response["cross_query"])

def process_collection(bm25_query, sem_query, cross_query):
    all = []
    for doc in collections:
        m = Model(doc)
        tp = len(fitz.open(doc))  # number of pages
        sections = m.assign_span_sections(tp)
        if not sections:
            sections = m.fallback_page_sections()
        non_empty = [s for s in sections if s["content"].strip()]
        if (len(non_empty) < 2):
            continue

        chunks = m.semantic_chunk_sections(sections)
        bm25 = m.build_bm25_index(chunks)
        bm25_res = m.bm25_retrieve(bm25, chunks, bm25_query)
        m.load_embedding_model()
        emb = m.build_embedding_index(chunks)
        sem_res = m.embedding_retrieve(emb, chunks, sem_query)
        candidates = merge(bm25_res[:50], sem_res[:50])
        m.load_cross_encoder()
        ranked_chunks = m.cross_encoder_rank(candidates, cross_query, persona, task)
        ranked_sections = m.aggregate_chunk_into_sections(ranked_chunks)

        for s in ranked_sections:
            s["document"] = Path(doc).stem
        all.extend(ranked_sections)

    all.sort(
        key=lambda x: x["importance_score"],
        reverse=True
    )

    for i, s in enumerate(all, start=1):
        s["importance_rank"] = i
    return all

all = process_collection(bm25_query, semantic_query, cross_query)

metadata = {
    "input_documents": [Path(d).stem for d in collections],
    "persona": persona,
    "job_to_be_done": task,
    "processing_timestamp": datetime.now().isoformat()
}

extracted_sections = []
for s in all[:5]:
    extracted_sections.append({
        "document": s["document"],
        "section_title": s["section_heading"],
        "importance_rank": s["importance_rank"],
        "page_number": s["page_start"] + 1,   # PDFs are 1-indexed
        "cross_score": s["cross_score"],
        "semantic_score": s["semantic_score"],
        "bm25_score": s["bm25_score"]
    })

def build_refined_text(section, max_chunks=2):
    chunks = sorted(
        section["chunks"],
        key=lambda x: x["cross_score"],
        reverse=True
    )[:max_chunks]

    texts = []
    for c in chunks:
        texts.append(c["text"].strip())

    return " ".join(texts)

subsection_analysis = []

for s in all[:5]:
    subsection_analysis.append({
        "document": s["document"],
        "refined_text": build_refined_text(s),
        "page_number": s["page_start"] + 1
    })

final_output = {
    "metadata": metadata,
    "extracted_sections": extracted_sections,
    "subsection_analysis": subsection_analysis
}

# evaluation of the output generated
def extract_eval_material(final_output):
    sections = final_output.get("extracted_sections", [])
    subsections = final_output.get("subsection_analysis", [])
    return sections, subsections

def relevance_agreement_score(extracted_sections):
    if not extracted_sections:
        return 0.0

    sem_scores = [s.get("semantic_score", 0.0) for s in extracted_sections]
    cross_scores = [s.get("cross_score", 0.0) for s in extracted_sections]
    bm25_scores = [s.get("bm25_score", 0.0) for s in extracted_sections]

    def min_max_norm(vals):
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    sem_norm = min_max_norm(sem_scores)
    cross_norm = min_max_norm(cross_scores)

    if max(bm25_scores) > 0:
        bm25_norm = min_max_norm(bm25_scores)
    else:
        bm25_norm = [0.0] * len(bm25_scores)

    scores = []

    for s, c, b in zip(sem_norm, cross_norm, bm25_norm):
        agreement = 1 - abs(s - c)

        strength = (0.45 * s) + (0.45 * c) + (0.10 * b)

        scores.append(
            0.6 * agreement +
            0.4 * strength
        )
    return sum(scores) / len(scores)



def job_alignment_score(embed_model, subsection_analysis, job):
    if not subsection_analysis:
        return 0.0

    job_emb = embed_model.encode(
        f"Task: {job}",
        normalize_embeddings=True
    )

    sims = []
    for s in subsection_analysis:
        text = s.get("refined_text", "").strip()
        if not text:
            continue

        text = text[:600]

        emb = embed_model.encode(text, normalize_embeddings=True)
        sims.append(float(emb @ job_emb))

    if not sims:
        return 0.0

    return sum(sims) / len(sims)


def coherence_penalty(subsections):
    penalty = 0.0

    for s in subsections:
        text = s["refined_text"].strip()

        if len(text.split()) < 40:
            penalty += 0.5
        if not text or not text[0].isupper():
            penalty += 0.5
        if text.count(".") < 2:
            penalty += 0.5

    return penalty / max(1, len(subsections))

def aggregate_scores(scores):
    return (
        0.41 * scores["relevance"] +
        0.32 * scores["alignment"] +
        0.27 * (1 - scores["coherence_penalty"])
    )

def evaluate_pipeline(final_output, job, embed_model=None):
    sections, subsections = extract_eval_material(final_output)
    scores = {}
    scores["relevance"] = relevance_agreement_score(sections)

    if embed_model:
        scores["alignment"] = job_alignment_score(embed_model, subsections, job)
    else:
        scores["alignment"] = 0.5  # neutral fallback

    scores["coherence_penalty"] = coherence_penalty(subsections)
    scores["final_score"] = aggregate_scores(scores)

    return scores

def filter_sections(sections, min_ratio=0.7, min_count=3):
    if not sections:
        return []

    best = sections[0]["importance_score"]
    filtered = [
        s for s in sections
        if s["importance_score"] >= min_ratio * best
    ]

    # safety fallback: keep top N
    if len(filtered) < min_count:
        filtered = sections[:min_count]

    return filtered

print(evaluate_pipeline(final_output, task, embed_model=SentenceTransformer("all-mpnet-base-v2")))


output_path = Path("./test/files/collection2/json/output.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)
