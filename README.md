# Gemini Question Processor

An automated tool that processes educational questions in Google Sheets using Google's Gemini AI. It cleans LaTeX/KaTeX formatting, normalizes text, and classifies questions by subject, topic, and difficulty level.

## What It Does

This script processes Google Sheets containing educational questions and performs two main tasks:

1. **Text Cleaning & Formatting**
   - Converts LaTeX/KaTeX math notation to human-readable Unicode (e.g., `\alpha` → α, `x^2` → x²)
   - Removes math wrappers like `\( \)`, `\[ \]`, `$$ $$`
   - Fixes split words caused by tabs/newlines
   - Preserves HTML tags (like `<img>` tags) for inline images
   - Normalizes whitespace while maintaining readability

2. **Question Classification**
   - **Subject**: Categorizes the question (Physics, Chemistry, Mathematics, Biology, etc.)
   - **Topic**: Identifies specific topics (e.g., "Newton's Laws", "Quadratic Equations")
   - **Difficulty Level**: Classifies as "Easy", "Medium", or "Hard"

The script updates the Google Sheet **in-place**, meaning it modifies the existing columns rather than creating new ones.

## Features

- ✅ Automatic credentials detection (`credentials.json` or `credential.json`)
- ✅ Environment variable support via `.env` file
- ✅ Batch processing for efficient API usage (default: 18 questions per batch)
- ✅ Image URL support (embeds images as HTML in corresponding text fields)
- ✅ Automatic retry logic with exponential backoff
- ✅ Preserves empty rows and handles missing data gracefully

## Prerequisites

- Python 3.7 or higher
- Google Cloud Service Account credentials (JSON file)
- Google Gemini API key
- A Google Sheet with the required columns

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd task5
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv env
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
env\Scripts\activate
```

**macOS/Linux:**
```bash
source env/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```


### 6. Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 7. Create `.env` File

Create a `.env` file in the project root with your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note:** Never commit the `.env` file or `credentials.json` to version control!

## Google Sheet Format

Your Google Sheet must have the following columns in the header row (row 1):

### Required Columns:
- `question_text` - The question text
- `option_a` - Option A
- `option_b` - Option B
- `option_c` - Option C
- `option_d` - Option D
- `explanation` - Explanation text
- `subject` - Will be filled by the script
- `topic` - Will be filled by the script
- `difficulty_level` - Will be filled by the script (Easy/Medium/Hard)

### Optional Columns (for image support):
- `question_image_url` - Image URL for the question
- `option_a_image_url` - Image URL for option A
- `option_b_image_url` - Image URL for option B
- `option_c_image_url` - Image URL for option C
- `option_d_image_url` - Image URL for option D
- `explanation_image_url` - Image URL for the explanation

If image URL columns exist, the script will automatically embed the images as HTML `<img>` tags in the corresponding text fields.

## Usage

1. Make sure your virtual environment is activated
2. Run the script:

```bash
python index.py
```

3. When prompted, paste your Google Sheet URL (must include `#gid=...`)

   Example URL format:
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=WORKSHEET_ID
   ```

4. The script will:
   - Automatically detect `credentials.json`
   - Load `GEMINI_API_KEY` from `.env`
   - Process all rows starting from row 2
   - Update the sheet in-place with cleaned text and classifications

## Configuration

You can modify these defaults in `index.py`:

```python
DEFAULT_MODEL = "gemini-2.5-flash"  # Gemini model to use
DEFAULT_BATCH_SIZE = 18              # Questions processed per API call
```

## How It Works

1. **Reads** questions from your Google Sheet in batches
2. **Sends** batches to Gemini AI for processing
3. **Receives** cleaned text and classifications
4. **Updates** the sheet in-place with the results
5. **Preserves** empty rows and handles errors gracefully

## Troubleshooting

### "Could not find credentials.json"
- Make sure `credentials.json` (or `credential.json`) is in the project root directory
- Check that the file name is spelled correctly

### "GEMINI_API_KEY not found"
- Verify your `.env` file exists in the project root
- Check that the `.env` file contains: `GEMINI_API_KEY=your_key_here`
- Make sure there are no extra spaces around the `=` sign

### "Could not open worksheet with gid=..."
- Verify the Google Sheet URL includes `#gid=XXXX`
- Ensure the service account email has been granted access to the sheet
- Check that the worksheet ID (gid) is correct

### "Missing required columns"
- Ensure all required columns exist in row 1 of your sheet
- Column names are case-sensitive and must match exactly

## Project Structure

```
task5/
├── index.py              # Main script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── credentials.json      # Google Service Account credentials (add this)
└── README.md            # This file
```

## Dependencies

Key dependencies:
- `google-genai` - Google Gemini AI client
- `gspread` - Google Sheets API wrapper
- `python-dotenv` - Environment variable management

See `requirements.txt` for the complete list.

## License

This project is provided as-is for educational and personal use.

