import os
import re
import fitz
import torch
import numpy as np
from docx import Document
import paddle
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FLAGS_use_cuda"] = "0"
os.environ["FLAGS_selected_gpus"] = "-1"
PROC_PAGE_LIMIT = 2

paddle.set_device('cpu')

use_gpu=False
paddle_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    show_log=False,
    use_gpu=use_gpu
)

def get_text_from_pdf_paddle(pdf_path: str) -> Tuple[str, str]:
    """
    Improved PDF text extraction that preserves tables and structure better
    """
    doc = fitz.open(pdf_path)
    full_text = []
    
    for page_num in range(len(doc)):
        if page_num + 1 > settings.process_page_limit:
            break
            
        page = doc.load_page(page_num)
        # First try regular text extraction
        text = page.get_text("text")
        
        # If text seems incomplete (e.g., missing tables), fall back to OCR
        if not text or len(text) < 100:  # Threshold for detecting bad extraction
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            ocr_results = paddle_ocr.ocr(img_np, cls=True)
            text = ' '.join([
                seg[1][0] 
                for line in ocr_results if line 
                for seg in line 
                if seg and len(seg) > 1 and len(seg[1]) > 0 and seg[1][0]
            ])
            
        full_text.append(text)

    combined_text = '\n\n'.join(full_text)
    
    # Save to file
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_file_path = os.path.join(settings.text_files_dir, f"{base_name}.txt")
    
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)
        
    return combined_text, txt_file_path

def get_text_from_pdf_paddle(pdf_path: str) -> Tuple[str, str]:
    doc = fitz.open(pdf_path)

    detected_text = []
    for page_num in tqdm(range(len(doc)), desc="Converting PDF pages to images"):
        if page_num + 1 > PROC_PAGE_LIMIT:
            break
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        if not text or len(text) < 100:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)
            ocr_results = paddle_ocr.ocr(img_np, cls=True)
            text = ' '.join([
                seg[1][0] 
                for line in ocr_results if line 
                for seg in line 
                if seg and len(seg) > 1 and len(seg[1]) > 0 and seg[1][0]
            ])
            
        detected_text.append(text)

    combined_text = '\n\n'.join(detected_text)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    return combined_text, txt_file_path

def get_text_from_image_paddle(image_path):

    ocr_results = paddle_ocr.ocr(image_path, cls=True)
    detected_text = [
        seg[1][0]
        for line in ocr_results if line  # skip None or empty lines
        for seg in line
        if seg and len(seg) > 1 and len(seg[1]) > 0 and seg[1][0]
    ]

    combined_text = ' '.join(detected_text)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[DEBUG] check output directory => ", output_dir)

    # Save combined text to file for testing....
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print("[DEBUG] check base name => ", base_name)

    txt_file_path = os.path.join(output_dir, f"{base_name}.txt")

    print("[DEBUG] check file => ", txt_file_path)

    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    return combined_text, txt_file_path


def get_text_from_word_paddle(word_path):
    """
    Read text from a .doc or .docx file with python-docx, save to a .txt,
    and return the text + path to the .txt.
    """

    doc = Document(word_path)
    full_text = [para.text for para in doc.paragraphs]

    combined_text = "\n".join(full_text)

    base_name = os.path.splitext(os.path.basename(word_path))[0]
    txt_file_path = os.path.join("../output", f"{base_name}.txt")

    with open(txt_file_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(combined_text)

    return combined_text, txt_file_path