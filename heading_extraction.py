import os
import json
import fitz
import numpy as np
import bisect
import itertools
from collections import defaultdict
from thefuzz import fuzz
import re
from itertools import groupby


class PDFOutline:
    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.elements = []
        self.lines = []
        self.features = {}
        self.heading_patterns = re.compile(
            r'^(abstract|introduction|summary|background|overview|conclusion|references|appendix|contents|figure|table|chapter|section|part|phase|postscript|preamble|glossary|index|acknowledgements|methodology|results|discussion|evaluation|approach|requirements|milestones|timeline|membership|terms of reference)\b|'
            r'^\d+(\.\d+)*\s+[A-Z]',
            re.IGNORECASE
        )

    @staticmethod
    def is_poster(lines_by_page, max_spans=80, max_pages=2):
        total_spans = sum(len(v) for v in lines_by_page.values())
        return total_spans <= max_spans or len(lines_by_page) <= max_pages
    
    @staticmethod
    def classify_poster(lines_by_page):
        all_lines = [line for lines in lines_by_page.values() for line in lines]
        if not all_lines:
            return [], ""
        sizes = [line["size"] for line in all_lines if line["text"]]
        if not sizes:
            return [], ""
        m, sd, md = np.mean(sizes), np.std(sizes), np.median(sizes)
        elements = []
        # Sort by page then y
        all_lines.sort(key=lambda x: (x["page"], round(x["bbox"].y1)))
        for (pg, y), group in groupby(all_lines, key=lambda x: (x["page"], round(x["bbox"].y1))):
            row = list(group)
            row = sorted(row, key=lambda x: x["bbox"].x0)
            txt = " ".join(r["text"] for r in row).strip()
            if not txt or re.fullmatch(r"\W+", txt):
                continue
            size = max(r["size"] for r in row)
            bold = any(r.get("font", "").lower().find("bold") != -1 for r in row)
            is_heading = size > m + 1.5*sd or (size > md * 1.2 and bold)
            elements.append({
                "text": txt,
                "page": pg,
                "level": "H1" if is_heading else "BODY_TEXT"
            })
        headings = [el for el in elements if el["level"] == "H1"]
        title = max(headings, key=lambda x: len(x["text"]))["text"] if headings else (elements[0]["text"] if elements else "")
        return [el for el in elements if el["level"].startswith("H")], title
    
    def is_semantic_heading(self, text):
        s = text.strip()
        return bool(self.heading_patterns.match(s) and len(s.split()) < 12)  #max number of words in heading < 12
    
    def build_lines_with_features(self):
        for pnum, page in enumerate(self.doc):
            width = page.rect.width
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
            for block in blocks:
                if block["type"] != 0: #skip image blocks
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    line_bbox = fitz.Rect(line["bbox"])
                    x_center = (line_bbox.x0 + line_bbox.x1) / 2
                    center_offset = abs((width / 2) - x_center)
                    horizontal_center = max(0, 1 - (center_offset / (width / 2)))
                    avg_size = sum(s["size"] for s in spans) / len(spans)
                    is_bold = any(s["flags"] & 16 for s in spans)
                    full_text = " ".join(s["text"] for s in spans).strip()
                    is_semantic = self.is_semantic_heading(full_text)
                    self.lines.append({
                        "text": full_text,
                        "size": avg_size,
                        "bold": is_bold,
                        "page": pnum,
                        "bbox": line_bbox,
                        "centeredness": horizontal_center,
                        "is_semantic": is_semantic
                    })
    
    def identify_headers_and_footers(self, top_n=5, bottom_n=3, threshold=75, tol=10):
        pages = defaultdict(list)
        for line in self.lines:
            pages[line["page"]].append(line)
        
        sorted_pages = {}
        for pnum, lines in pages.items():
            lines.sort(key = lambda l: l["bbox"].y0)
            sorted_pages[pnum] = {
                "lines" : lines,
                "y_coords": [l["bbox"].y0 for l in lines]
            }

        candidates = set()
        for pnum in range(1, len(self.doc)):
            prev = sorted_pages.get(pnum - 1)
            curr = sorted_pages.get(pnum)
            if not prev or not curr:
                continue
            def match_lines(curr_lines, prev_data):
                for ln in curr_lines:
                    y = ln["bbox"].y0
                    lo = bisect.bisect_left(prev_data["y_coords"], y - tol)
                    hi = bisect.bisect_right(prev_data["y_coords"], y + tol)
                    for i in range(lo, hi):
                        pl = prev_data["lines"][i]
                        if fuzz.ratio(ln["text"], pl["text"]) > threshold:
                            candidates.add((ln["text"], int(y)))
                            candidates.add((pl["text"], int(pl["bbox"].y0)))
                            break
            match_lines(curr["lines"][:top_n], prev)
            match_lines(curr["lines"][-bottom_n:], prev)
        return candidates
    
    def get_statistical_features(self):
        #to calculate the mean heading size ignore the headings with very large size as they skew the statistics
        sizes = [l["size"] for l in self.lines if l["size"] < 18]
        arr = np.array(sizes) if sizes else np.array([12])
        self.features = {
            "mean_size": float(arr.mean()),
            "std_dev_size": float(arr.std()),
            "median_size": float(np.median(arr))
        }
    
    def classify_as_headings(self):
        m, sd, md = (self.features[k] for k in ("mean_size", "std_dev_size", "median_size"))
        for l in self.lines:
            is_heading = (
                (l["is_semantic"] and (l["bold"] or l["size"] > md * 1.05)) or
                (l["bold"] and l["size"] > md * 1.15) or
                (l["size"] > m + 1.8 * sd)
            )
            if is_heading and len(l["text"].split()) <= 20:
                self.elements.append({**l, "cls": "HEADING"})

    def assign_headings_by_level(self, el):
         # generalised heuristic to assign H1–H4 based on style
        size, bold, centered = el["size"], el["bold"], el["centeredness"]
        m, sd, md = (self.features[k] for k in ("mean_size", "std_dev_size", "median_size"))
        if size > m + 1.5 * sd and bold: return "H1"
        if size > m + 1.0 * sd and centered > 0.7: return "H1"
        if bold and size > m: return "H2"
        if bold or (el["is_semantic"] and size > md): return "H3"
        return "H4"
    
    def create_json(self):
        #get the title from document's first page
        page0 = [l for l in self.lines if l["page"] == 0]
        candidates = sorted(page0, key=lambda l: (l["size"], len(l["text"])), reverse=True)
        title = next((c["text"].strip() for c in candidates if len(c["text"]) > 5), "Untitled Document")
        title_set = {line["text"] for line in candidates[:2]}
        heads = [e for e in self.elements if e["text"] not in title_set]

        outline = []
        for h in heads:
            level = self.assign_headings_by_level(h)
            outline.append({"level": level, "text": h["text"], "page": h["page"]})

        # sort by page and level, remove duplicates
        order = {"H1": 1, "H2": 2, "H3": 3, "H4": 4}
        outline.sort(key=lambda x: (x["page"], order.get(x["level"], 99)))
        final = [next(g) for _, g in itertools.groupby(outline, key=lambda x: (x["text"], x["page"]))]
        return {"title": title, "outline": final}

    def analyse(self):
        self.build_lines_with_features()
        lines_by_page = defaultdict(list)
        for l in self.lines:
            lines_by_page[l["page"]].append(l)
        if self.is_poster(lines_by_page):
            poster_headings, title = self.classify_poster(lines_by_page)
            outline = []
            for el in poster_headings:
                # Try to assign level based on size (since poster mode is flat, everything H1)
                outline.append({"level": el.get("level", "H1"), "text": el["text"], "page": el["page"]})
            return lines_by_page, json.dumps({"title": title, "outline": outline}, indent=4)
        
        # normal documents
        header = self.identify_headers_and_footers()
        self.lines = [l for l in self.lines if (l["text"], int(l["bbox"].y0)) not in header]
        self.get_statistical_features()
        self.classify_as_headings()
        return lines_by_page, json.dumps(self.create_json(), indent=4)
    
# pdf_path = 
# lines_by_page, outline_json = PDFOutline(pdf_path).analyse()     # returns JSON string
# outline = json.loads(outline_json)                     # parse to dict if needed
# print(outline_json)


