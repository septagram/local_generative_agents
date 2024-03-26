import openai
from typing import List
from transformers import AutoModel
from langchain_core.embeddings import Embeddings

from utils import *

embedding_model_instances = {}

def get_openai_embedding(text, model=embedding_model):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']

def get_local_embedding(text, model=embedding_model):
  if model not in embedding_model_instances:
    embedding_model_instances[model] = AutoModel.from_pretrained(model, trust_remote_code=True) # trust_remote_code is needed to use the encode method
  embeddings = embedding_model_instances[model].encode([text])
  return embeddings[0]

if embedding_is_local:
  get_embedding = get_local_embedding
else:
  get_embedding = get_openai_embedding

class LocalEmbeddings(Embeddings):
  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return [self.embed_query(item) for item in texts]

  def embed_query(self, text: str) -> List[float]:
    return get_embedding(text).tolist()
