import os
import pymupdf4llm

PDF_PATH = os.path.join("data", "HS_code_Nomenclature.pdf")
OUTPUT_PATH = os.path.splitext(PDF_PATH)[0] + ".md"


def convert_pdf_to_markdown(pdf_path: str, output_path: str) -> None:
    markdown_text = pymupdf4llm.to_markdown(pdf_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(markdown_text)

    print(f"Markdown 파일 생성 완료: {output_path}")


if __name__ == "__main__":
    convert_pdf_to_markdown(PDF_PATH, OUTPUT_PATH)