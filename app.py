# 1. Install libraries
# pip install pypdf sentence-transformers faiss-cpu openai

# 2. Import libraries
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import os

# 3. Load PDF
pdf = PdfReader("college_handbook.pdf")
text = "\n".join(page.extract_text() or "" for page in pdf.pages)

# 4. Split into chunks
chunk_size = 500
overlap = 100

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size - overlap)
    if text[i:i + chunk_size].strip()
]

# 5. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    chunks, normalize_embeddings=True
).astype("float32")

# 6. Create FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 7. Ask a question
question = input("Ask a question about the document: ")

# 8. Retrieve relevant chunks
query_embedding = model.encode(
    [question], normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_embedding, k=3)
context = "\n\n".join(
    chunks[i] for i in indices[0] if i != -1
)

# 9. Generate an answer
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using only the supplied document context. "
                "If the answer is not present, say you cannot find it."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
)

# 10. Display the answer
print("\nAnswer:", response.choices[0].message.content)
