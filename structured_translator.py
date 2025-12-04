"""
Generic Structured PDF Translator.

This module provides a robust solution for translating PDF documents from English to Spanish
while strictly preserving the original layout, formatting, and font styles. It utilizes
Google Cloud Translation API and PyMuPDF (fitz) for PDF manipulation.

Key Features:
- Strict layout preservation using bounding box constraints.
- Automatic font scaling and text wrapping.
- Language detection to translate only English content.
- Production-ready error handling and logging.
"""

import argparse
import logging
import os
import re
import sys
from typing import List, Tuple, Optional

import fitz
from dotenv import load_dotenv
from google.cloud import translate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional dependency: langdetect
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logger.warning("langdetect library not found. Language filtering will rely on heuristics.")


class TranslationConfig:
    """Handles configuration and environment setup."""
    
    def __init__(self):
        load_dotenv()
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.region = os.getenv("GCP_REGION")
        self.credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not self.project_id:
            logger.warning("GCP_PROJECT_ID not found in environment variables.")


class TextProcessor:
    """Handles text analysis, language detection, and structure preservation."""
    
    def is_english(self, text: str) -> bool:
        """Determines if the provided text is English."""
        if not text or not text.strip():
            return False
            
        # Heuristic: Skip if mostly numbers or symbols
        cleaned = re.sub(r'[0-9\W]', '', text)
        if len(cleaned) < 2:
            return False
            
        # Heuristic: Skip single short words
        if len(text.split()) == 1 and len(text) < 4:
            return False

        if HAS_LANGDETECT:
            try:
                lang = detect(text)
                return lang == 'en'
            except Exception:
                return False 
        
        # Fallback heuristic
        common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 
                        'is', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this'}
        words = set(text.lower().split())
        return len(words.intersection(common_words)) > 0

    def preserve_structure(self, original: str, translated: str) -> str:
        """Preserves leading/trailing whitespace and numbering prefixes."""
        if not translated:
            return translated

        leading_ws = len(original) - len(original.lstrip())
        trailing_ws = len(original) - len(original.rstrip())
        
        core_orig = original.strip()
        core_trans = translated.strip()
        
        # Regex for numbering: "1.", "1.2", "A.", "(a)", "1)"
        pattern = r'^(\d+(?:\.\d+)*[\.\)]?|[A-Za-z][\.\)])\s*'
        orig_match = re.match(pattern, core_orig)
        trans_match = re.match(pattern, core_trans)
        
        if orig_match:
            prefix = orig_match.group(1)
            if not trans_match or trans_match.group(1) != prefix:
                if trans_match:
                    core_trans = core_trans[trans_match.end():].lstrip()
                core_trans = f"{prefix} {core_trans}"
        
        return (" " * leading_ws) + core_trans + (" " * trailing_ws)


class LayoutEngine:
    """
    Manages text layout to ensure translated text fits within original bounding boxes.
    """
    
    def __init__(self):
        self.symbol_fonts = ['wingdings', 'symbol', 'zapfdingbats']

    def get_font_name(self, original_font: str, flags: int) -> Tuple[str, bool, bool]:
        """Maps PDF font to a standard Base14 font."""
        font_lower = original_font.lower()
        
        for sym in self.symbol_fonts:
            if sym in font_lower:
                return sym, False, False

        is_bold = bool(flags & 16) or 'bold' in font_lower or 'black' in font_lower
        is_italic = bool(flags & 64) or 'italic' in font_lower or 'oblique' in font_lower
        
        if 'times' in font_lower or 'serif' in font_lower:
            base = 'times'
        elif 'courier' in font_lower or 'mono' in font_lower:
            base = 'cour'
        else:
            base = 'helv'
            
        return base, is_bold, is_italic

    def calculate_fit(self, page, text: str, bbox: fitz.Rect, font_base: str, 
                      original_size: float, is_bold: bool, is_italic: bool) -> Tuple[float, List[str], str]:
        """
        Calculates optimal font size and wrapping to fit text within bbox.
        """
        suffix = ""
        if is_bold and is_italic: suffix = "-boldoblique" if font_base == 'helv' else "-bolditalic"
        elif is_bold: suffix = "-bold"
        elif is_italic: suffix = "-oblique" if font_base == 'helv' else "-italic"
        font_name = font_base + suffix
        
        try:
            font = fitz.Font(font_name)
        except Exception:
            font = fitz.Font("helv")
            font_name = "helv"

        # Strategy 1: Fit on one line
        width = font.text_length(text, fontsize=original_size)
        if width <= bbox.width:
            return original_size, [text], font_name
            
        scale_width = bbox.width / width
        new_size_width = original_size * scale_width
        
        if new_size_width > original_size * 0.6:
             return new_size_width, [text], font_name

        # Strategy 2: Wrap text
        lines = []
        words = text.split()
        current_line = ""
        
        test_size = original_size * 0.9 
        min_size = 4.0
        
        while test_size >= min_size:
            lines = []
            current_line = ""
            valid_wrap = True
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if font.text_length(test_line, fontsize=test_size) <= bbox.width:
                    current_line = test_line
                else:
                    if current_line: lines.append(current_line)
                    current_line = word
                    if font.text_length(word, fontsize=test_size) > bbox.width:
                        valid_wrap = False
                        break
            if current_line: lines.append(current_line)
            
            if not valid_wrap:
                test_size -= 0.5
                continue
                
            line_height = test_size * 1.2
            total_height = len(lines) * line_height
            
            if total_height <= bbox.height + 2:
                return test_size, lines, font_name
            
            test_size -= 0.5

        return max(new_size_width, min_size), [text], font_name


class GenericPDFTranslator:
    """Main application class for PDF translation."""
    
    def __init__(self):
        self.config = TranslationConfig()
        self.processor = TextProcessor()
        self.layout = LayoutEngine()
        self.client = None
        
        try:
            self.client = translate.TranslationServiceClient()
            self.parent = f"projects/{self.config.project_id}/locations/{self.config.region}"
        except Exception as e:
            logger.error(f"Failed to initialize Google Translate client: {e}")

    def translate_text(self, text: str) -> str:
        """Translates a single text string."""
        if not self.client or not text:
            return text
            
        try:
            response = self.client.translate_text(
                request={
                    "parent": self.parent,
                    "contents": [text],
                    "source_language_code": "en",
                    "target_language_code": "es",
                    "mime_type": "text/plain"
                }
            )
            return response.translations[0].translated_text
        except Exception as e:
            logger.error(f"Translation API error: {e}")
            return text

    def process_file(self, input_path: str, output_path: str):
        """Processes a single PDF file."""
        logger.info(f"Processing file: {input_path}")
        
        try:
            doc = fitz.open(input_path)
        except Exception as e:
            logger.error(f"Could not open file {input_path}: {e}")
            return

        for page_num, page in enumerate(doc):
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")
            
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            
            spans = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            spans.append(span)
            
            # Filter English text
            spans_to_translate = [s for s in spans if self.processor.is_english(s["text"].strip())]
            
            if not spans_to_translate:
                continue
                
            for span in spans_to_translate:
                original_text = span["text"]
                bbox = fitz.Rect(span["bbox"])
                
                translated_text = self.translate_text(original_text)
                translated_text = self.processor.preserve_structure(original_text, translated_text)
                
                if translated_text == original_text:
                    continue
                    
                # Redact original
                page.add_redact_annot(bbox)
                page.apply_redactions()
                
                # Layout calculation
                font_base, is_bold, is_italic = self.layout.get_font_name(span["font"], span["flags"])
                
                font_size, lines, font_name = self.layout.calculate_fit(
                    page, translated_text, bbox, font_base, span["size"], is_bold, is_italic
                )
                
                # Insert translated text
                color = span["color"]
                if isinstance(color, int):
                    b = (color & 255) / 255
                    g = ((color >> 8) & 255) / 255
                    r = ((color >> 16) & 255) / 255
                    color = (r, g, b)
                
                try:
                    page.insert_textbox(
                        bbox, 
                        "\n".join(lines), 
                        fontsize=font_size, 
                        fontname=font_name, 
                        color=color,
                        align=0
                    )
                except Exception as e:
                    logger.warning(f"Text insertion failed on page {page_num + 1}: {e}")

        try:
            doc.save(output_path, garbage=4, deflate=True)
            logger.info(f"Successfully saved translated file to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Translate PDF documents while preserving layout.")
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF file or directory.")
    parser.add_argument("--output", "-o", required=True, help="Path to output PDF file or directory.")
    
    args = parser.parse_args()
    
    translator = GenericPDFTranslator()
    
    if os.path.isfile(args.input):
        # Single file mode
        if os.path.isdir(args.output):
            filename = os.path.basename(args.input)
            output_path = os.path.join(args.output, f"translated_{filename}")
        else:
            output_path = args.output
            
        translator.process_file(args.input, output_path)
        
    elif os.path.isdir(args.input):
        # Directory mode
        if not os.path.exists(args.output):
            os.makedirs(args.output)
            
        for filename in os.listdir(args.input):
            if filename.lower().endswith(".pdf"):
                input_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, f"translated_{filename}")
                translator.process_file(input_path, output_path)
    else:
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()
