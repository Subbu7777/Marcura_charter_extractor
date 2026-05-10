# Charter Party Clause Extractor

Extract legal clauses from maritime Charter Party PDF documents using Google Gemini LLM.

## Prerequisites

- **Python 3.10+**
- **Google Gemini API Key** — Get one at: https://aistudio.google.com/apikey

## Quick Start

### Setup

1. Create and activate virtual environment:
   ```bash
   # Create
   python -m venv venv_name
   
   # Activate (Windows PowerShell)
   .\venv_name\Scripts\Activate.ps1
   
   # Activate (Windows Command Prompt)
   .\venv_name\Scripts\activate.bat
   
   # Activate (Linux/macOS)
   source venv_name/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your values:
   GEMINI_MODEL = your_gemini_model_name
   GEMINI_API_KEY = your_key_here
   ```

### Basic Usage

```bash
# Extract from default voyage charter example (pages 6-39)
python main.py

# Custom output path
python main.py --output clauses.json

# Custom page range
python main.py --pages 6 39  #should start with 6

# Use a custom PDF URL
python main.py --url https://example.com/charter.pdf

# Use a local PDF file if you have one in local
python main.py --pdf local_file.pdf

# Enable debug logging
python main.py --verbose
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pdf` | None | Local PDF file path (overrides `--url`) |
| `--url` | voyage-charter-example | PDF URL to download |
| `--pages START END` | `6 39` | Page range (1-indexed, inclusive) |
| `--output, -o` | `extracted_clauses.json` | Output JSON file path |
| `--verbose, -v` | False | Enable debug logging |

## Output Format

```json
{
  "clauses": [
    {
      "id": "1",
      "title": "Condition Of Vessel",
      "text": "The vessel shall be seaworthy, properly manned, equipped and supplied..."
    }
  ]
}
```

## How It Works

1. **PDF Input** — Download from URL or read local file
2. **Text Extraction** — Extract text using PyMuPDF, filter strikethrough text
3. **Chunking** — Split into 200K-character chunks at page boundaries
4. **LLM Extraction** — Send each chunk to Gemini with structured prompts
5. **Deduplication** — Merge clauses spanning chunks, remove duplicates
6. **JSON Output** — Save structured results

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GEMINI_API_KEY not set` | Create `.env` file with your API key |
| `GEMINI_MODEL not set` | Add model name to `.env` file |
| `API key not valid` | Verify key at https://aistudio.google.com/apikey |
| `No text extracted` | Check page range; PDF may be scanned images |
| Missing clauses | Use `--verbose` to debug; check page boundaries |
| Truncated text | Strikethrough detection may be too aggressive |


## Dependencies

| Package | Purpose |
|---------|---------|
| `google-genai` | Google Gemini API SDK |
| `PyMuPDF` | PDF text extraction & vector graphics analysis |
| `pydantic` | Data validation & JSON serialization |
| `requests` | HTTP downloads |
| `python-dotenv` | Environment variable management |

