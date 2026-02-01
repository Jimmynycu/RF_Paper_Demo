import pdfplumber
import os

# PDF file paths
pdfs = {
    "paper1_chu_limit": r"D:\download\Chu-Limit-Guided_Decomposition-Based_Multiobjective_Large-Scale_Optimization_for_Generative_Broadband_Electrically_Small_Antenna_Design.pdf",
    "paper2_pinn_fss": r"D:\download\2024_CN_Inverse Design of Frequency Selective Surface Using Physics-Informed Neural Networks.pdf",
    "paper3_llm_rimsa": r"D:\download\2025_CN_LLM-RIMSA_Large Language Models driven Reconfigurable Intelligent Metasurface Antenna Systems.pdf"
}

for name, pdf_path in pdfs.items():
    print(f"Processing: {name}")
    output_file = f"{name}.txt"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {os.path.basename(pdf_path)}\n")
                f.write(f"# Total Pages: {len(pdf.pages)}\n\n")
                
                for j, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        f.write(f"\n=== PAGE {j} ===\n\n")
                        f.write(text)
                        f.write("\n")
        print(f"  Saved to: {output_file}")
    except Exception as e:
        print(f"  Error: {e}")
