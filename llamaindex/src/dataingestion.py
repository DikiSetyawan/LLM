import os
import logging
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(
    filename='process.log',  # Log file
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def convert_pdf_to_txt(pdf_filepath, output_dir):
    """
    Convert a PDF file to a text file and store it in the specified directory.
    
    :param pdf_filepath: Path to the PDF file to be converted.
    :param output_dir: Directory where the output .txt file will be stored.
    """
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Create the output .txt file path
        base_name = os.path.basename(pdf_filepath)
        txt_filename = os.path.splitext(base_name)[0] + ".txt"
        txt_filepath = os.path.join(output_dir, txt_filename)

        # Read the PDF file
        reader = PdfReader(pdf_filepath)
        logging.info(f"Opened PDF file: {pdf_filepath}")

        # Extract text from the PDF
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += page.extract_text()
            logging.info(f"Extracted text from page {page_num + 1}")

        # Write the text to the .txt file
        with open(txt_filepath, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
        logging.info(f"Converted {pdf_filepath} to {txt_filepath}")

    except Exception as e:
        logging.error(f"Failed to convert {pdf_filepath} to txt: {e}")