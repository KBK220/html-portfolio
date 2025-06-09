# html-portfolio

This repository now includes a simple Python script to fetch job listings based on keywords extracted from a resume. The script will search Indeed, Naukri and LinkedIn using publicly available job search pages.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key (optional, improves query generation):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Run the script with your resume PDF:
   ```bash
   python job_agent.py <path to resume.pdf> --location "City"
   ```
   The script prints the top results from each site.

Make sure you comply with each website's terms of service when scraping.
