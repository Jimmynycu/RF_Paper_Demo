import pdfplumber
import os

# PDF file paths
pdfs = [
    r"D:\download\Chu-Limit-Guided_Decomposition-Based_Multiobjective_Large-Scale_Optimization_for_Generative_Broadband_Electrically_Small_Antenna_Design.pdf",
    r"D:\download\2024_CN_Inverse Design of Frequency Selective Surface Using Physics-Informed Neural Networks.pdf",
    r"D:\download\2025_CN_LLM-RIMSA_Large Language Models driven Reconfigurable Intelligent Metasurface Antenna Systems.pdf"
]

for i, pdf_path in enumerate(pdfs, 1):
    print(f"\n{'='*80}")
    print(f"PAPER {i}: {os.path.basename(pdf_path)}")
    print(f"{'='*80}\n")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for j, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    print(f"\n--- Page {j} ---\n")
                    print(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
    
    print(f"\n{'='*80}")
    print(f"END OF PAPER {i}")
    print(f"{'='*80}\n")
