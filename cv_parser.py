import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import os
import cv2
import numpy as np
from difflib import get_close_matches

# Configure Tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Standard CV sections to look for
STANDARD_SECTIONS = [
    "ABOUT ME", "SUMMARY", "PROFILE", "OBJECTIVE",
    "EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY",
    "EDUCATION", "ACADEMIC BACKGROUND", "QUALIFICATIONS",
    "SKILLS", "TECHNICAL SKILLS", "SOFT SKILLS", "LANGUAGE SKILLS",
    "PROJECTS", "RESEARCH PROJECTS",
    "CERTIFICATIONS", "TRAINING", "COURSES",
    "LANGUAGES", "LANGUAGE PROFICIENCY",
    "CONTACT", "CONTACT INFORMATION", "PERSONAL DETAILS",
    "INTERESTS", "HOBBIES", "ADDITIONAL ACTIVITIES",
    "VOLUNTEERING", "VOLUNTEER WORK",
    "AWARDS", "ACHIEVEMENTS", "HONORS",
    "PUBLICATIONS", "RESEARCH PAPERS",
    "REFERENCES"
]

def preprocess_image(img):
    """Enhance image for better OCR results"""
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    return denoised

def extract_text_from_image(img):
    """Extract text from image with enhanced OCR"""
    processed_img = preprocess_image(img)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(processed_img, config=custom_config)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with fallback to OCR"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                text += extract_text_from_image(img)
    return text

def extract_text_from_image_file(image_path):
    """Extract text from image files"""
    img = Image.open(image_path)
    return extract_text_from_image(img)

def normalize_section_header(header):
    """Match extracted headers to standard sections"""
    header_upper = header.upper().strip()
    
    # Try exact match first
    if header_upper in STANDARD_SECTIONS:
        return header_upper
    
    # Try close matches
    matches = get_close_matches(header_upper, STANDARD_SECTIONS, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    
    # Special cases
    if "WORK" in header_upper and "HISTORY" in header_upper:
        return "WORK EXPERIENCE"
    if "EDUCATION" in header_upper or "QUALIFICATION" in header_upper:
        return "EDUCATION"
    if "CONTACT" in header_upper or "DETAILS" in header_upper:
        return "CONTACT"
    
    return None

def identify_sections(text):
    """Identify and extract CV sections"""
    sections = {}
    current_section = "GENERAL INFORMATION"
    sections[current_section] = []
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # Check if line could be a section header
        is_header = False
        normalized_header = normalize_section_header(line)
        
        # Header detection criteria
        if normalized_header:
            is_header = True
        elif line.isupper() and len(line.split()) <= 5:
            is_header = True
        elif re.match(r'^[A-Z][a-z]+( [A-Z][a-z]+)*$', line) and len(line.split()) <= 4:
            is_header = True
        
        # If header found, start new section
        if is_header:
            section_name = normalized_header if normalized_header else line.upper()
            current_section = section_name
            if current_section not in sections:
                sections[current_section] = []
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
    
    return sections
def extract_education(text):
    """Extract education details"""
    education_list = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    current_edu = {}
    for line in lines:
        # Detect degree and institution
        if re.search(r'(Bachelor|Master|PhD|B\.Sc|M\.Sc|Diploma|Degree|Licence|Engineering)', line, re.IGNORECASE):
            if current_edu:
                education_list.append(current_edu)
                current_edu = {}
            current_edu["degree"] = line
        
        # Detect university or school name
        elif re.search(r'(University|School|College|Institute)', line, re.IGNORECASE):
            current_edu["institution"] = line
        
        # Detect date
        elif re.search(r'(\d{4})\s?[-–—]\s?(\d{4}|Present)', line):
            current_edu["date"] = line
    
    if current_edu:
        education_list.append(current_edu)
    
    return education_list
def extract_languages(text):
    """Extract languages from text"""
    languages = re.split(r'[,•\n]', text)
    return [lang.strip() for lang in languages if lang.strip()]

def extract_structured_data(sections):
    """Extract structured data from categorized sections"""
    structured_data = {}
    
    # Process each section
    for section, content in sections.items():
        content_text = '\n'.join(content)
        
        # Special handling for different sections
        if section in ["CONTACT", "CONTACT INFORMATION"]:
            structured_data["contact"] = extract_contact_info(content_text)
        elif section in ["SKILLS", "TECHNICAL SKILLS", "SOFT SKILLS"]:
            structured_data["skills"] = extract_skills(content_text)
        elif section in ["EXPERIENCE", "WORK EXPERIENCE"]:
            structured_data["experience"] = extract_experience(content_text)
        elif section == "EDUCATION":
            structured_data["education"] = extract_education(content_text)
        elif section == "LANGUAGES":
            structured_data["languages"] = extract_languages(content_text)
        else:
            structured_data[section.lower().replace(" ", "_")] = content_text
    
    return structured_data

def extract_contact_info(text):
    """Extract contact information with enhanced patterns"""
    info = {
        "name": None,
        "email": None,
        "phone": [],
        "linkedin": None,
        "address": None,
        "website": None
    }
    
    # Name detection (first line that looks like a name)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:5]:
        if (line.istitle() or line.isupper()) and 1 < len(line.split()) <= 4:
            info["name"] = line
            break
    
    # Email
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if emails:
        info["email"] = emails[0]
    
    # Phone (multiple formats)
    phones = re.findall(r'(\+?\d[\d\s\-\(\)]{7,}\d)', text)
    info["phone"] = [phone.strip() for phone in phones]
    
    # LinkedIn
    linkedin = re.search(r'(linkedin\.com/[^\s]+)', text, re.I)
    if linkedin:
        info["linkedin"] = "https://" + linkedin.group(1)
    
    # Address (simple pattern)
    address = re.search(r'(\d{1,5}\s[\w\s]{3,},\s[\w\s]{3,},\s[A-Z]{2}\s\d{5})', text)
    if address:
        info["address"] = address.group(1)
    
    return info

def extract_skills(text):
    """Extract skills from skills section"""
    skills = []
    
    # Split by commas, bullets, or newlines
    items = re.split(r'[,•\n]', text)
    
    for item in items:
        item = item.strip()
        if item and len(item.split()) <= 5:  # Skip long paragraphs
            skills.append(item)
    
    return skills

def extract_experience(text):
    """Extract structured experience data"""
    experiences = []
    current_exp = {}
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        # Detect job title pattern
        if re.match(r'^[A-Z][a-z]+( [A-Z][a-z]+)*$', line) and len(line.split()) <= 5:
            if current_exp:  # Save previous experience
                experiences.append(current_exp)
                current_exp = {}
            current_exp["title"] = line
        
        # Detect company and date pattern
        elif re.match(r'^[A-Z][a-zA-Z0-9& ]+,\s[A-Z][a-z]+\s\d{4}\s?[-–—]\s?(\w+\s\d{4}|Present)', line):
            parts = re.split(r',\s|\s[-–—]\s', line)
            if len(parts) >= 3:
                current_exp["company"] = parts[0]
                current_exp["date"] = ' '.join(parts[1:])
        
        # Bullet points (responsibilities)
        elif line.startswith('- ') or re.match(r'^•\s', line):
            if "responsibilities" not in current_exp:
                current_exp["responsibilities"] = []
            current_exp["responsibilities"].append(line[2:].strip())
    
    if current_exp:
        experiences.append(current_exp)
    
    return experiences

def process_file(file_path):
    """Process any supported file type"""
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Extract text based on file type
        if file_ext == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_ext in ('.png', '.jpg', '.jpeg'):
            text = extract_text_from_image_file(file_path)
        else:
            return {"error": "Unsupported file type"}
        
        if not text.strip():
            return {"error": "No text could be extracted"}
        
        # Identify and extract sections
        sections = identify_sections(text)
        structured_data = extract_structured_data(sections)
        
        return structured_data
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    file_path = r"C:\Users\R I B\OneDrive\Images\Hadilcv\cvHadil.pdf"
    
    print(f"🔍 Processing file: {file_path}")
    result = process_file(file_path)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("\n✅ Structured CV Data:")
        for section, content in result.items():
            print(f"\n=== {section.upper().replace('_', ' ')} ===")
            if isinstance(content, dict):
                for k, v in content.items():
                    print(f"  • {k.title()}: {v}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            print(f"  • {k.title()}: {v}")
                        print()
                    else:
                        print(f"  • {item}")
            else:
                print(content)