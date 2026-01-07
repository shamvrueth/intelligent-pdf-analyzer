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

class Model:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def get_page_word_boxes(self):
        doc = fitz.open(self.pdf_path)
        box = {}
        for p in range(len(doc)):
            page = doc[p]
            words = page.get_text("words")
            words = sorted(words, key=lambda w: (round(w[1],2), round(w[0],2)))
            box[p] = words
        return box
    
    def assign_span_sections(self, tp):
        sections = []
        lines_by_page, outline_json = PDFOutline(self.pdf_path).analyse()     # returns JSON string
        print(outline_json)
        outline = json.loads(outline_json)
        total_pages = tp - 1
        n = len(outline["outline"])
        l = []
        k = 0

        # add line index to lines_by_page
        for p in lines_by_page:
            for i, d in enumerate(lines_by_page[p]):
                d["index"] = i

        # add line index and in page index to outline
        for i in range(n - 1):
            if (outline["outline"][i]["page"] not in l):
                k = 0
                l.append(outline["outline"][i]["page"])
            outline["outline"][i]["in_page_index"] = k
            k += 1
            
            for j in lines_by_page[outline["outline"][i]["page"]]:
                if (j["text"] == outline["outline"][i]["text"]):
                    outline["outline"][i]["start_index"] = j["index"]
                if (j["text"] == outline["outline"][i+1]["text"]):
                    outline["outline"][i]["end_index"] = j["index"] - 1

        print(outline)
        if (outline["outline"][n-1]["page"] not in l):
            outline["outline"][n-1]["in_page_index"] = 0
        else:
            outline["outline"][n-1]["in_page_index"] = k
        outline["outline"][n-1]["start_index"] = outline["outline"][n-2]["end_index"]
        outline["outline"][n-1]["end_index"] = len(lines_by_page[outline["outline"][n-1]["page"]]) - 1
        
        for i, h in enumerate(outline["outline"]):
            s = {}
            s["heading"] = h["text"]
            s["level"] = h["level"]
            s["start_page"] = h["page"]
            if i + 1 < n:
                next_h = outline["outline"][i + 1]
                if next_h["page"] == h["page"]:
                    s["end_page"] = h["page"]
                    s["end_after_in_page_index"] = next_h.get("in_page_index")
                    s["end_index"] = next_h.get("start_index") - 1
                else:
                    s["end_page"] = next_h["page"]
                    s["end_index"] = next_h.get("start_index") - 1
            else:
                s["end_page"] = total_pages
            s["start_index"] = h["start_index"]
            sections.append(s)

        for s in sections:
            start = s["start_page"]
            end = s["end_page"]
            if start == end and "end_after_in_page_index" in s and "end_index" in s:
                lines = ""
                for i in lines_by_page[start]:
                    if (s["start_index"] > i["index"]):
                        continue
                    if (i["index"] > s["end_index"]):
                        break
                    lines += " " + i["text"]
                s["content"] = lines
            elif "end_index" not in s:
                t = []
                lines = ""
                for i in range(start, end + 1):
                    for j in lines_by_page[i]:
                        if (i == start and s["start_index"] > j["index"]):
                            continue
                        lines += " " + j["text"]
                    t.append(lines)
                s["content"] = "\n".join(t)
            else:
                t = []
                lines = ""
                for i in range(start, end + 1):
                    for j in lines_by_page[i]:
                        if (i == start and s["start_index"] > j["index"]):
                            continue
                        if (i == end and j["index"] > s["end_index"]):
                            break
                        lines += " " + j["text"]
                    t.append(lines)
                s["content"] = "\n".join(t)
        
        return sections
    
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
                    "section_level": s["level"],
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
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25
    
    def bm25_retrieve(self, bm25, chunks, query, top_k=200):
        query = query.lower()
        query = re.sub(r"[^\w\s]", " ", query)
        query_tokens = query.split()
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **chunks[idx],
                "bm25_score": float(scores[idx])
            })

        return results
    
    def load_embedding_model(self, model_name="all-mpnet-base-v2"):
        self.embed_model = SentenceTransformer(model_name)
    
    def build_embedding_index(self, chunks, batch_size=32):
        texts = [c["text"] for c in chunks]

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
    
    def cross_encoder_rank(self, candidates, query, top_k=20, batch_size=8):
        scores = []

        with torch.no_grad():
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]

                pairs = [(query, f"Section: {c['section_heading']}. Content: {c['text']}")
                        for c in batch]

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
                        "cross_score": float(s)
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
            sections.append(
                {
                    "section_heading": heading,
                    # "section_level": chunks["section_level"],
                    "page_start": ps,
                    "page_end": pe,
                    "importance_score": imp,
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
        r.setdefault("bm25_score", 0.0)
        r.setdefault("semantic_score", 0.0)
    return list(merged.values())

# m = Model("./test/files/collection/South of France - Cuisine.pdf")
# sections = m.assign_span_sections(14)
# chunks = m.overlapping_chunk_sections(sections)
# bm25 = m.build_bm25_index(chunks)
# results = m.bm25_retrieve(bm25, chunks, "Le Louis XV")
# m.load_embedding_model()
# embeddings = m.build_embedding_index(chunks)
# res = m.embedding_retrieve(embeddings, chunks, "Le Louis XV")
# candidates = merge(results, res)
# m.load_cross_encoder()
# final = m.cross_encoder_rank(candidates, "Le Louis XV")
# s = m.aggregate_chunk_into_sections(final)
# print(s[:3])

