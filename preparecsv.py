import csv
import json
from pathlib import Path
from heading_extraction import PDFOutline

class PrepareCSV:
    def __init__(self, pdf_path, output_path):
        # self.outline = PDFOutline(pdf_path)
        self.pdf_path = pdf_path
        self.output_path = output_path
        p1 = Path(pdf_path)
        l1 = list(p1.glob("*.pdf"))
        print(f"Found {len(l1)} PDFs in {p1}")
        # p2 = Path(output_path)
        # l2 = list(p2.glob("*.json"))
        for i in range(len(l1)):
            self.outline = PDFOutline(l1[i])
            self.prepare_input_csv()
            # self.prepare_output_csv(l2[i])
            
    def prepare_output_csv(self, json_file):
        pdf_name = Path(json_file).stem
        csv_file_out = "output.csv"
        with open(json_file, "r") as f:
            output = json.load(f)

        fields = ["id", "text", "page", "level"]
        
        write_header_out = not Path(csv_file_out).exists() or Path(csv_file_out).stat().st_size == 0
        with open(csv_file_out, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if write_header_out:
                writer.writeheader()
            writer.writerow({"id": pdf_name, "text": output["title"], "page": 0, "level": "H1"})
            for i in output["outline"]:
                writer.writerow({"id": pdf_name, "text": i.get("text"), "page": i.get("page"), "level": i.get("level")})

    def determine_label(self, text, outline_json, pno):
        outline = json.loads(outline_json)
        title = outline["title"]
        if title == text:
            return "HEADING"
        for heading in outline["outline"]:
            if heading["text"] == text and pno == heading["page"]:
                return "HEADING"
        return "NONE"

    def prepare_input_csv(self):
        pdf_name = Path(self.outline.pdf_path).stem
        lines_by_page, outline_json = self.outline.analyse()
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / f"{pdf_name}.csv"
        write_header = not Path(csv_file).exists() or Path(csv_file).stat().st_size == 0

        fieldnames = ["id", "text", "font_size", "is_bold", "page", "x0", "y0", "x1", "y1", "centeredness", "is_semantic", "label"]
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for pnum, lines in lines_by_page.items():
                for line in lines:
                    writer.writerow({
                        "id": pdf_name,
                        "text": line.get("text"),
                        "font_size": line.get("size"),
                        "is_bold": line.get("bold"),
                        "page": line.get("page"),
                        "x0": line["bbox"].x0,
                        "y0": line["bbox"].y0,
                        "x1": line["bbox"].x1,
                        "y1": line["bbox"].y1,
                        "centeredness": line.get("centeredness"),
                        "is_semantic": line.get("is_semantic"),
                        "label": self.determine_label(line.get("text"), outline_json, line.get("page"))
                    })



# pdf_path = "./sample_dataset/files"
# output_path = "./sample_dataset/outputs"

# PrepareCSV(pdf_path, output_path)