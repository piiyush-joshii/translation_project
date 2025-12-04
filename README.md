# PDF Layout-Preserving Translator

A robust, production-ready Python tool for translating PDF documents from English to Spanish while strictly preserving the original layout, formatting, and font styles.

## Features

*   **Strict Layout Preservation**: Uses a "shrink-to-fit" algorithm to ensure translated text stays within the original bounding boxes, preventing overlaps.
*   **Smart Language Detection**: Automatically detects English text to avoid translating numbers, codes, or existing Spanish content.
*   **Font Matching**: Maps PDF fonts to standard compatible fonts (Helvetica, Times, Courier) while preserving Bold and Italic styles.
*   **Batch Processing**: Supports processing single files or entire directories.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd translation_project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `langdetect` separately if not in requirements:*
    ```bash
    pip install langdetect
    ```

3.  **Set up Google Cloud Credentials**:
    *   Create a `.env` file in the root directory.
    *   Add your Google Cloud Project ID and Region:
        ```env
        GCP_PROJECT_ID=your-project-id
        GCP_REGION=your-region
        GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
        ```

## Usage

### Translate a Single File
```bash
python structured_translator.py --input path/to/input.pdf --output path/to/output.pdf
```

### Translate a Directory
```bash
python structured_translator.py --input ./input_folder --output ./output_folder
```

## Requirements

*   Python 3.8+
*   `google-cloud-translate`
*   `pymupdf` (fitz)
*   `python-dotenv`
*   `langdetect` (optional, but recommended)

## License

[Your License Here]
