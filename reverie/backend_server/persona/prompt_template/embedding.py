"""
 Copyright 2024 Igor Novikov

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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
