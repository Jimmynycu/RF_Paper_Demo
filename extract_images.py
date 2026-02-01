# /// script
# requires-python = ">=3.10"
# dependencies = ["pdfplumber", "pillow", "pymupdf"]
# ///

import fitz  # PyMuPDF
import os

# Create images directory
os.makedirs('paper_images', exist_ok=True)

papers = [
    ('paper1_chu_limit_antenna.pdf', 'p1'),
    ('paper2_pinn_fss.pdf', 'p2'),
    ('paper3_llm_rimsa.pdf', 'p3')
]

for pdf_name, prefix in papers:
    try:
        doc = fitz.open(pdf_name)
        img_count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images(full=True)
            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                img_filename = f"paper_images/{prefix}_page{page_num+1}_img{img_idx+1}.{image_ext}"
                with open(img_filename, "wb") as f:
                    f.write(image_bytes)
                img_count += 1
        
        print(f"{pdf_name}: Extracted {img_count} images from {len(doc)} pages")
        doc.close()
    except Exception as e:
        print(f"{pdf_name}: Error - {e}")

print(f"\nImages saved to paper_images/ directory")
