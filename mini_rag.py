"""
mini_rag.py - a tiny RAG (retrieval-augmented generation) prototype.

What it does:
  1. Loads small text files from a ./docs folder (creates 3 fake sample manuals if missing).
  2. Splits them into chunks.
  3. RETRIEVE: finds the chunks most similar to your question (TF-IDF similarity).
  4. AUGMENT + GENERATE: sends only those chunks plus your question to an LLM through its API.

Setup (once):
  pip install anthropic scikit-learn
  export ANTHROPIC_API_KEY="your-key"        (Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key")

Run:
  python mini_rag.py

NOTE: Use only fake/public text in ./docs. Never put confidential or personal data into an LLM.
"""
import os
import glob
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "claude-sonnet-5-5"   # change if your account lists a different model name
DOCS_DIR = "docs"
TOP_K = 3                      # how many chunks to retrieve per question

SAMPLE_DOCS = {
    "payroll_app_runbook.txt": (
        "Payroll App Runbook\n\n"
        "Restarting the service: log in to the app server, run the restart script, then confirm the "
        "health page returns OK. If the health page is not OK after 5 minutes, open a P2 incident.\n\n"
        "Monthly maintenance window: the first Sunday of each month from 2am to 4am. Notify "
        "stakeholders by email 5 business days before the window.\n"
    ),
    "reporting_app_manual.txt": (
        "Reporting App Manual\n\n"
        "Nightly report job: runs at 1am and writes results to the reports database. If the job fails, "
        "check the job log for Oracle connection errors first, then rerun the job manually.\n\n"
        "Password rotation: service account passwords are rotated every 90 days. After rotation, "
        "update the secret in the vault and restart the reporting service.\n"
    ),
    "incident_process.txt": (
        "Incident Process\n\n"
        "P1 incidents: service is down for all users. Page the on-call engineer immediately and "
        "notify the IT supervisor. P2 incidents: partial outage or degraded performance. "
        "Respond within 1 hour.\n\n"
        "After every P1 or P2, hold a review within 5 business days to find the root cause "
        "and record the lesson learned in the knowledge base.\n"
    ),
}


def ensure_sample_docs():
    os.makedirs(DOCS_DIR, exist_ok=True)
    if not glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        for name, text in SAMPLE_DOCS.items():
            with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
                f.write(text)
        print(f"Created sample documents in ./{DOCS_DIR}")


def load_chunks():
    """Split each file into paragraph-sized chunks, remembering the source file."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            for para in f.read().split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"source": os.path.basename(path), "text": para})
    return chunks


def retrieve(question, chunks, vectorizer, chunk_vectors, k=TOP_K):
    """Return the k chunks most similar to the question."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, chunk_vectors)[0]
    best = scores.argsort()[::-1][:k]
    return [chunks[i] for i in best if scores[i] > 0]


def answer(client, question, retrieved):
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in retrieved)
    system = (
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Cite the source file name in brackets after each fact."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    ensure_sample_docs()
    chunks = load_chunks()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    chunk_vectors = vectorizer.fit_transform([c["text"] for c in chunks])
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    print(f"Loaded {len(chunks)} chunks. Ask a question (or type 'quit').")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        retrieved = retrieve(question, chunks, vectorizer, chunk_vectors)
        print("\nRetrieved:", [c["source"] for c in retrieved])
        print("\nAnswer:", answer(client, question, retrieved))


if __name__ == "__main__":
    main()
