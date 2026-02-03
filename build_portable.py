
import re
import base64
import os
import mimetypes
from pathlib import Path

# Configuration
SOURCE_FILE = 'demo_dashboard.html'
OUTPUT_FILE = 'RF_AI_Portfolio_PORTABLE.html'
BASE_DIR = os.getcwd()

def get_mime_type(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type
    # Fallback
    ext = filename.lower().split('.')[-1]
    if ext == 'png': return 'image/png'
    if ext in ['jpg', 'jpeg']: return 'image/jpeg'
    if ext == 'svg': return 'image/svg+xml'
    return 'application/octet-stream'

def image_to_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = get_mime_type(path)
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def main():
    print(f"Reading {SOURCE_FILE}...")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Regex to find img src attributes
    # simplified regex, assumes src="path" matches
    img_regex = r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>'
    
    matches = re.finditer(img_regex, html_content)
    
    replacements = {}
    
    print("Finding and encoding images...")
    for match in matches:
        img_path_str = match.group(1)
        
        # Skip if already base64 or http link
        if img_path_str.startswith('data:') or img_path_str.startswith('http'):
            continue
            
        full_path = os.path.join(BASE_DIR, img_path_str)
        
        if os.path.exists(full_path):
            print(f"Encoding: {img_path_str}")
            b64_data = image_to_base64(full_path)
            if b64_data:
                replacements[img_path_str] = b64_data
        else:
            print(f"Warning: Image not found: {full_path}")

    # Perform replacements
    new_html_content = html_content
    for path, data in replacements.items():
        # Use simple replace for safety, ensuring we match the quote style roughly or just the string
        # To be safe against partial matches, we typically want to be careful, but these paths are likely unique enough.
        new_html_content = new_html_content.replace(path, data)

    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_html_content)
        
    print("Done!")

if __name__ == "__main__":
    main()
