#!/usr/bin/env bash

rm -rf .venv/
uv venv
uv add fastembed flagembedding qdrant-client sentence-transformers
source .venv/bin/activate

docker-compose up -d
sleep 30

python embed_documents.py

docker-compose down -v

rm -rf .venv/