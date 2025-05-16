import json
import faiss
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

from sklearn.preprocessing import normalize

# Load model, index, metadata
model = SentenceTransformer('all-mpnet-base-v2')

index_blogs = faiss.read_index("mynachiketa_blogs.faiss")
index_books = faiss.read_index('blogs-book.faiss')

with open("mynachiketa_blogs_metadata.json", "r", encoding="utf-8") as f:
    blogs = json.load(f)
    
with open('books_metadata.json', 'r', encoding='utf-8')as f:
    books = json.load(f)
    

#FASTAPI SETUP
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

# baseModel - Blogs
class BlogPageDetails_blogs(BaseModel):
    title: str
    language: str
    category: str
    author: str
    read_time: str

# baseModel - Books
class BlogPageDetails_books(BaseModel):
    title: str
    language: str
    category: str
    author: str

@app.api_route('/', methods=['GET', 'HEAD', 'POST'])
def read_root():
    return {"msg": "Recommendation API is running!"}

@app.get('/health')
def health():
    return {"status": "healthy"}


@app.post('/recommendBlogs')
def recommend(req: BlogPageDetails_blogs):
    # step-01: Create a query string for embedding
    query_text = (
        f"{req.title}"
        f"language: {req.language} "
        f"category: {req.category} "
        f"author: {req.author} "
        f"read_time: {req.read_time} minutes"
    )
    
    # step-2: Generate embedding
    embedding = model.encode([query_text])
    
    # step-3: Normalize embedding (if your FAISS index was built on normalized vectors)
    embedding = normalize(embedding)
    
    # step-4: Search FAISS index
    top_k = 6
    distances, indices = index_blogs.search(embedding.astype(np.float32), top_k)
    
    # step-5: fetch results from metadata
    results = [blogs[i] for i in indices[0]]
    
    return {"results": results[1:]}  # Skip first result assuming it's the query blog itself



@app.post('/recommendBooks')
def book_recommend(req: BlogPageDetails_books):
    # step-01 construct search query
    query_text = (
        f"{req.title}. This blog is about {req.category}, written in {req.language} by {req.author}"
    )
    
    # step-02: Get embedding for query
    query_embedding = model.encode([query_text])
    
    # step-03: Normalize embedding (if your FAISS index was built on normalized vectors)
    query_embedding = normalize(query_embedding)
    
    # step-04: search faiss index
    top_k = 5
    distances, indices = index_books.search(query_embedding.astype(np.float32), top_k)
    
    # step-05: fetch results from metadata
    results = [books[i] for i in indices[0]]
    
    return {"results": results}