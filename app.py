from fastapi import FastAPI, APIRouter

app = FastAPI()

embedding_router = APIRouter()

@embedding_router.post("/embeddings")
async def generate_embeddings(text: str):
    import torch
    from src.embedding_code import generate_embedding
    embedding = generate_embedding(text)
    return {"embedding": embedding, "generated_by": "embedding_model"}

app.include_router(embedding_router)