import os
import requests
import openai
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from pathlib import Path
import xml.etree.ElementTree as ET
# === CONFIG ===
client = openai.api_key = os.getenv("OPENAI_API_KEY")

DATABASE_FOLDER = "Database"
Path(DATABASE_FOLDER).mkdir(exist_ok=True)

# === EMBEDDING ===
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# === SEARCH & PARSE ARXIV ===
def fetch_arxiv_metadata(query, max_results=20):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{query.replace(" ", "+")}&start=0&max_results={max_results}'
    response = requests.get(base_url + search_query)
    
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code}")
        return []

    root = ET.fromstring(response.content)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    results = []
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text.strip()
        abstract = entry.find('atom:summary', namespace).text.strip()
        link = entry.find('atom:id', namespace).text.strip()
        arxiv_id = link.split('/abs/')[-1]
        results.append({'title': title, 'link': link, 'arxiv_id': arxiv_id, 'abstract': abstract})
    
    return results

# === RANKING BY SEMANTIC SIMILARITY ===
def find_best_match(user_query, papers):
    print("🔎 Getting embeddings and comparing similarity...")
    query_embedding = get_embedding(user_query)
    scored = []

    for paper in papers:
        abstract = paper['abstract'][:1000]
        paper_embedding = get_embedding(abstract)
        similarity = cosine_similarity([query_embedding], [paper_embedding])[0][0]
        scored.append((similarity, paper))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]  # Best match

# === PDF DOWNLOAD ===
def download_pdf(arxiv_id):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    local_path = os.path.join(DATABASE_FOLDER, f"{arxiv_id}.pdf")
    response = requests.get(url)

    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        return local_path
    else:
        raise Exception(f"Failed to download PDF for {arxiv_id}")

# === READ PDF TEXT ===
def read_pdf_text(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === SUMMARIZE ===
def summarize_text(text, model="gpt-4", max_tokens=700):
    print("🧠 Summarizing...")
    prompt = f"Summarize the following research paper content in a concise and informative way:\n\n{text[:4000]}"
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# === MAIN PIPELINE ===
def semantic_search_and_summarize(query):
    print(f"🎯 Searching for papers on: {query}")
    papers = fetch_arxiv_metadata(query)

    if not papers:
        print("❌ No papers found.")
        return None

    best_paper = find_best_match(query, papers)

    print(f"\n✅ Best Match: {best_paper['title']}")
    print(f"🔗 Link: {best_paper['link']}")
    print("⬇️ Downloading PDF...")
    pdf_path = download_pdf(best_paper['arxiv_id'])

    print("📖 Reading PDF...")
    full_text = read_pdf_text(pdf_path)

    print("📝 Generating summary...")
    summary = summarize_text(full_text)

    return {
        "title": best_paper['title'],
        "summary": summary,
        "pdf_path": pdf_path,
        "link": best_paper['link']
    }

# === ENTRY POINT ===
if __name__ == "__main__":
    user_query = input("🔍 Enter a research topic: ")
    result = semantic_search_and_summarize(user_query)

    if result:
        print("\n=== SUMMARY ===")
        print(f"📄 Title: {result['title']}")
        print(f"📁 Saved PDF: {result['pdf_path']}")
        print(f"🔗 Link: {result['link']}")
        print(f"\n🧠 Summary:\n{result['summary']}")
