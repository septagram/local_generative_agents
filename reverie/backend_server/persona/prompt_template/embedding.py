from transformers import AutoModel

from gpt_structure import get_openai_embedding
from utils import *

embedding_model_instances = {}

def get_local_embedding(text, model=embedding_model):
  if model not in embedding_model_instances:
    embedding_model_instances[model] = AutoModel.from_pretrained(model, trust_remote_code=True) # trust_remote_code is needed to use the encode method
  embeddings = embedding_model_instances[model].encode([text])
  return embeddings[0]

if embedding_is_local:
  get_embedding = get_local_embedding
else:
  get_embedding = get_openai_embedding
