import os
import requests
from bs4 import BeautifulSoup
import pdfminer.high_level
import re
from collections import Counter

try:
    import openai
except ImportError:  # pragma: no cover - optional dependency
    openai = None

STOPWORDS = set([
    'the','and','to','of','in','a','for','on','with','is','that','by','as','at','an','be','are','from','this','or','it'
])

def extract_text(resume_path):
    """Extract text from a PDF or plain text resume."""
    if resume_path.lower().endswith(".pdf"):
        return pdfminer.high_level.extract_text(resume_path)
    with open(resume_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_keywords(text, top_n=10):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    most_common = Counter(words).most_common(top_n)
    return [w for w, _ in most_common]

def openai_query(text):
    """Generate a search query using OpenAI if API key is configured."""
    if not openai:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
    prompt = (
        "Extract 3 to 4 concise job search keywords from this resume text. "
        "Return a comma-separated list.\n" + text[:2000]
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
        )
        if resp and resp.choices:
            return resp.choices[0].message["content"].strip()
    except Exception:
        return None

def search_indeed(query, location=""):
    params = {"q": query}
    if location:
        params["l"] = location
    response = requests.get("https://www.indeed.com/jobs", params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []
    for card in soup.select(".result"):
        title = card.select_one("h2 span")
        if title:
            job = {
                "title": title.text.strip(),
                "link": "https://www.indeed.com" + card.get("href", "")
            }
            jobs.append(job)
    return jobs

def search_naukri(query, location=""):
    url_query = query.replace(" ", "-")
    url_location = location.replace(" ", "-")
    url = f"https://www.naukri.com/{url_query}-jobs-in-{url_location}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []
    for card in soup.select("article.jobTuple"):
        title = card.select_one("a.title")
        if title:
            jobs.append({"title": title.text.strip(), "link": title.get("href")})
    return jobs

def search_linkedin(query, location=""):
    params = {"keywords": query}
    if location:
        params["location"] = location
    response = requests.get("https://www.linkedin.com/jobs/search/", params=params)
    soup = BeautifulSoup(response.text, "html.parser")
    jobs = []
    for card in soup.select("div.base-card"):
        title = card.select_one("h3")
        link = card.select_one("a.base-card__full-link")
        if title and link:
            jobs.append({"title": title.text.strip(), "link": link.get("href")})
    return jobs

def find_jobs(resume_pdf, location=""):
    text = extract_text(resume_pdf)
    query = openai_query(text)
    if not query:
        keywords = extract_keywords(text)
        query = " ".join(keywords[:3])
    print(f"Searching for jobs with query: {query}")
    indeed_jobs = search_indeed(query, location)
    naukri_jobs = search_naukri(query, location)
    linkedin_jobs = search_linkedin(query, location)
    return {
        "Indeed": indeed_jobs,
        "Naukri": naukri_jobs,
        "LinkedIn": linkedin_jobs
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch jobs based on resume keywords")
    parser.add_argument("resume", help="Path to resume PDF")
    parser.add_argument("--location", default="", help="Job location")
    args = parser.parse_args()
    results = find_jobs(args.resume, args.location)
    for site, jobs in results.items():
        print(f"\n{site} jobs:")
        for job in jobs[:5]:
            print(f"- {job['title']}: {job['link']}")

