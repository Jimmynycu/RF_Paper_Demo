import fitz  # PyMuPDF
import pdfplumber
import os
from PIL import Image
import io

# PDF file paths
papers = [
    ('paper1_chu_limit_antenna.pdf', 'p1'),
    ('paper2_pinn_fss.pdf', 'p2'),
    ('paper3_llm_rimsa.pdf', 'p3')
]

# Output directory
output_dir = 'paper_images/new_extraction'
os.makedirs(output_dir, exist_ok=True)

def extract_content(pdf_path, prefix):
    print(f"Processing {pdf_path}...")
    
    # 1. Extract Images using PyMuPDF (fitz)
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Filter out very small images (often icons or artifacts)
                if len(image_bytes) < 1000: # Skip images smaller than 1KB
                     continue

                image_name = f"{prefix}_p{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_name)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                print(f"  Extracted image: {image_name}")
    except Exception as e:
        print(f"  Error extracting images from {pdf_path}: {e}")

    # 2. Extract Tables as Snapshots using pdfplumber & PyMuPDF
    try:
        with pdfplumber.open(pdf_path) as pdf:
            doc = fitz.open(pdf_path) # Re-open with fitz for high-quality snapshots
            for page_num, page in enumerate(pdf.pages):
                tables = page.find_tables()
                if tables:
                    fitz_page = doc[page_num]
                    for table_index, table in enumerate(tables):
                        bbox = table.bbox
                        # bbox is (x0, top, x1, bottom)
                        
                        # Add a small padding
                        pad = 5
                        rect = fitz.Rect(bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad)
                        
                        # Render high-res image of the table area
                        pix = fitz_page.get_pixmap(clip=rect, dpi=300)
                        
                        table_name = f"{prefix}_p{page_num+1}_table{table_index+1}.png"
                        table_path = os.path.join(output_dir, table_name)
                        
                        pix.save(table_path)
                        print(f"  Extracted table snapshot: {table_name}")
    except Exception as e:
        print(f"  Error extracting tables from {pdf_path}: {e}")

    print(f"Finished processing {pdf_path}.\n")

if __name__ == "__main__":
    for pdf_file, prefix in papers:
        if os.path.exists(pdf_file):
            extract_content(pdf_file, prefix)
        else:
            print(f"File not found: {pdf_file}")

    print(f"All extractions saved to {output_dir}")
