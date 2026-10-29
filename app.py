"""
===============================================================
TRADITIONAL RAG PROJECT: COLLEGE HANDBOOK QUESTION ANSWERING
===============================================================

What this program does:
1. Reads a synthetic college handbook PDF.
2. Splits the handbook text into smaller chunks.
3. Converts each chunk into an embedding (a numerical vector).
4. Stores those vectors in a FAISS index.
5. Converts a student's question into an embedding.
6. Retrieves the most relevant chunks using semantic similarity.
7. Builds a prompt using the question and retrieved context.
8. Sends the prompt to an LLM (OpenAI) to generate an answer.

The college handbook is fictional and is only for learning.

Install required libraries in your terminal or notebook:
    pip install pypdf sentence-transformers faiss-cpu google-genai numpy

Before running the answer-generation part, set your API key securely:
    macOS/Linux:
        export GEMINI_API_KEY="your-api-key"

    Windows PowerShell:
        $env:GEMINI_API_KEY="your-api-key"

Do not share your API key or commit it to a public code repository.
"""

# ---------------------------------------------------------------
# STEP 0: IMPORT LIBRARIES
# ---------------------------------------------------------------

import os                       # Reads environment variables, including the API key
from pathlib import Path        # Helps us work with file paths
import numpy as np              # Handles numerical arrays
import faiss                    # Performs fast vector similarity search
from pypdf import PdfReader     # Reads text from PDF files
from sentence_transformers import SentenceTransformer  # Creates embeddings
from google import genai        # Official Google Gemini SDK for answer generation


# ---------------------------------------------------------------
# STEP 1: CONFIGURATION
# ---------------------------------------------------------------

# Keep the PDF in the same folder as this Python script.
# Change this name if your PDF has a different filename.
PDF_PATH = Path("sample_college_handbook.pdf")

# Number of chunks to retrieve for each question.
# For this simple demo, we retrieve the 3 most similar chunks.
TOP_K = 3

# Name of a small, commonly used sentence embedding model.
# The first run may download the model from the model hub.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# The OpenAI model used only for generating the final answer.
# You need a valid API key and API access to use this step.
GENERATION_MODEL_NAME = "gemini-3.8-flash"  # Change if your API account uses another available model


# ---------------------------------------------------------------
# STEP 2: LOAD THE PDF AND EXTRACT ITS TEXT
# ---------------------------------------------------------------

def load_pdf_text(pdf_path: Path) -> str:
    """
    Open the PDF and collect text from every page.

    Why?
    The embedding model works on text. We first need to extract the
    text contained in the college handbook PDF.

    Note:
    pypdf extracts selectable text. If the PDF contains only scanned
    images, you may need OCR (optical character recognition).
    """

    # Stop with a clear message if the file path is incorrect.
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Could not find: {pdf_path.resolve()}\n"
            "Place sample_college_handbook.pdf in the same folder "
            "as this script, or update PDF_PATH."
        )

    # Open the PDF.
    reader = PdfReader(str(pdf_path))

    # Store extracted text from each page in this list.
    page_texts = []

    # Visit each page one by one.
    for page_number, page in enumerate(reader.pages, start=1):
        # Extract text; use an empty string if a page has no extractable text.
        extracted_text = page.extract_text() or ""

        # Add a page label so we can see where text came from.
        page_texts.append(
            f"\n[Page {page_number}]\n{extracted_text}"
        )

    # Join all page text into one long string.
    full_text = "\n".join(page_texts)

    # A basic check to catch empty/scanned PDFs.
    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from the PDF. "
            "It may be scanned and require OCR."
        )

    print(f"PDF loaded: {pdf_path.name}")
    print(f"Pages read: {len(reader.pages)}")
    print(f"Characters extracted: {len(full_text)}")

    return full_text


# ---------------------------------------------------------------
# STEP 3: CHUNKING — SPLIT THE TEXT INTO SMALLER PIECES
# ---------------------------------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = 700,
    overlap: int = 120
) -> list[str]:
    """
    Split a long text into overlapping character-based chunks.

    chunk_size:
        Maximum approximate number of characters in each chunk.

    overlap:
        Number of characters repeated between adjacent chunks.

    Why overlap?
    A rule may begin at the end of one chunk and finish in the next.
    Overlap helps preserve some context across chunk boundaries.

    Teaching note:
    This is a simple character-based approach. More advanced systems
    split by paragraphs, headings, sentences, or token counts.
    """

    # Replace repeated whitespace and line breaks with single spaces.
    cleaned_text = " ".join(text.split())

    # Prevent invalid settings that could cause an infinite loop.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    chunks = []
    start = 0

    # Move through the document until we reach the end.
    while start < len(cleaned_text):
        # Pick the end position of this chunk.
        end = start + chunk_size

        # Extract the text between start and end.
        chunk = cleaned_text[start:end].strip()

        # Do not store empty chunks.
        if chunk:
            chunks.append(chunk)

        # If we reached the end, there is nothing more to split.
        if end >= len(cleaned_text):
            break

        # Start the next chunk before the previous one ended.
        # This creates the overlap.
        start = end - overlap

    print(f"Chunks created: {len(chunks)}")
    return chunks


# ---------------------------------------------------------------
# STEP 4: CREATE EMBEDDINGS FOR THE DOCUMENT CHUNKS
# ---------------------------------------------------------------

def create_embeddings(chunks: list[str], embedding_model):
    """
    Convert each text chunk into a numerical vector.

    An embedding represents text in a numerical space. Texts with
    related meanings may have vectors that are close to each other.

    This is not the same as generating an answer with an LLM.
    The embedding model is used for semantic search.
    """

    # Encode all chunks in one call.
    # normalize_embeddings=True makes each vector unit length.
    vectors = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # FAISS expects a 2D array of 32-bit floating-point numbers.
    vectors = np.asarray(vectors, dtype="float32")

    print(f"Embedding matrix shape: {vectors.shape}")
    # Example: (number_of_chunks, embedding_dimensions)

    return vectors


# ---------------------------------------------------------------
# STEP 5: BUILD A FAISS VECTOR INDEX
# ---------------------------------------------------------------

def build_faiss_index(embedding_vectors):
    """
    Store vectors in a FAISS index for similarity search.

    IndexFlatIP means:
        - Flat: compare against all stored vectors (simple exact search).
        - IP: inner product.

    Because the embeddings are normalized, inner product ranking is
    equivalent to cosine-similarity ranking.
    """

    # Each vector has a fixed number of dimensions.
    vector_dimensions = embedding_vectors.shape[1]

    # Create an empty index that compares vectors using inner product.
    vector_index = faiss.IndexFlatIP(vector_dimensions)

    # Add all document chunk vectors to the index.
    vector_index.add(embedding_vectors)

    print(f"Vectors stored in FAISS: {vector_index.ntotal}")

    return vector_index


# ---------------------------------------------------------------
# STEP 6: SEMANTIC SEARCH — RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------------

def retrieve_relevant_chunks(
    question: str,
    chunks: list[str],
    embedding_model,
    vector_index,
    top_k: int = TOP_K
):
    """
    Find the document chunks most similar in meaning to the question.

    Retrieval process:
        1. Embed the student's question.
        2. Search the FAISS index.
        3. Return the matching chunks and similarity scores.

    No generative LLM is needed for this retrieval step.
    """

    # Convert the question to a vector using the SAME embedding model
    # used for the document chunks.
    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # FAISS expects float32 vectors.
    question_vector = np.asarray(question_vector, dtype="float32")

    # Do not ask FAISS for more chunks than exist in the index.
    number_to_retrieve = min(top_k, len(chunks))

    # Search the index.
    # scores: similarity scores for each match.
    # ids: positions of matching chunks in the chunks list.
    scores, ids = vector_index.search(
        question_vector,
        number_to_retrieve
    )

    retrieved = []

    # The search result contains one row because we asked one question.
    for score, chunk_id in zip(scores[0], ids[0]):
        # FAISS uses -1 if there is no result for a requested position.
        if chunk_id == -1:
            continue

        retrieved.append({
            "chunk_id": int(chunk_id),
            "score": float(score),
            "text": chunks[int(chunk_id)]
        })

    return retrieved


# ---------------------------------------------------------------
# STEP 7: BUILD THE PROMPT
# ---------------------------------------------------------------

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Combine the retrieved document text and student's question.

    The retrieved context is called 'augmented context'.
    The LLM will receive this context alongside the question.

    Important:
    A prompt instruction helps guide the model, but it does not
    guarantee that every answer will be correct.
    """

    # Join the retrieved chunks into a single context block.
    context = "\n\n".join(
        f"Retrieved chunk {item['chunk_id']}:\n{item['text']}"
        for item in retrieved_chunks
    )

    # This is the prompt template.
    prompt = f"""
You are an assistant answering questions about a fictional college handbook.

Instructions:
- Use only the handbook context provided below.
- Do not invent rules, percentages, deadlines, or dates.
- If the context does not contain the answer, say:
  "I couldn't find that information in the sample handbook."
- Explain the answer clearly and briefly.
- Remember that this is a synthetic teaching document, not an official policy.

HANDBOOK CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    return prompt


# ---------------------------------------------------------------
# STEP 8: GENERATE THE ANSWER WITH AN LLM
# ---------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    """
    Send the prompt to the Gemini API and return the generated answer.

    This is the GENERATION part of Retrieval-Augmented Generation.
    It is separate from the embedding model and FAISS retrieval.
    """

    # Read the Gemini API key from an environment variable.
    # This avoids putting a secret key directly in the source code.
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY was not found. Set it in your environment "
            "before using answer generation."
        )

    # Create a Gemini API client using your key.
    client = genai.Client(api_key=api_key)

    # Send the prompt to the selected Gemini model.
    response = client.models.generate_content(
        model=GENERATION_MODEL_NAME,
        contents=prompt
    )

    # Extract the generated text from Gemini's response.
    return response.text or ""


# ---------------------------------------------------------------
# STEP 9: CONNECT THE STEPS INTO ONE RAG PIPELINE
# ---------------------------------------------------------------

def main():
    """
    Run the full pipeline and allow the student to ask questions.
    """

    # A. Read the handbook PDF.
    handbook_text = load_pdf_text(PDF_PATH)

    # B. Split the text into chunks.
    chunks = split_into_chunks(
        handbook_text,
        chunk_size=700,
        overlap=120
    )

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # C. Load the embedding model.
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # D. Convert all chunks to embeddings.
    print("\nCreating document embeddings...")
    document_vectors = create_embeddings(chunks, embedding_model)

    # E. Add the embeddings to the FAISS index.
    print("\nBuilding FAISS vector index...")
    vector_index = build_faiss_index(document_vectors)

    print("\nRAG system is ready!")
    print("Example question: How many classes can I miss and still sit the exam?")
    print("Type 'exit' to stop.\n")

    # F. Keep accepting questions until the user exits.
    while True:
        question = input("Student question: ").strip()

        # Stop the loop if the user types exit or quit.
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Ignore empty input.
        if not question:
            print("Please enter a question.")
            continue

        # G. Retrieve the most relevant document chunks.
        retrieved_chunks = retrieve_relevant_chunks(
            question=question,
            chunks=chunks,
            embedding_model=embedding_model,
            vector_index=vector_index,
            top_k=TOP_K
        )

        # Show the retrieved results so students can see what RAG found.
        print("\n--- Retrieved chunks (semantic search) ---")
        for result in retrieved_chunks:
            print(
                f"\nChunk ID: {result['chunk_id']} | "
                f"Similarity score: {result['score']:.3f}"
            )
            print(result["text"])

        # H. Construct the prompt from the question and retrieved chunks.
        prompt = build_prompt(question, retrieved_chunks)

        print("\n--- Prompt sent to the LLM ---")
        print(prompt)

        # I. Generate a final answer using the LLM.
        try:
            answer = generate_answer(prompt)
            print("\n--- RAG answer ---")
            print(answer)

        except Exception as error:
            # Retrieval has already worked even if the LLM/API call fails.
            print("\nAnswer generation failed.")
            print("Check your Gemini API key, model availability, and internet connection.")
            print(f"Error details: {error}")

        print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------
# STEP 10: START THE PROGRAM
# ---------------------------------------------------------------

# This condition means main() runs when this file is executed directly.
if __name__ == "__main__":
    main()
