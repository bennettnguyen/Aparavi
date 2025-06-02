# Enterprise RAG System

To run:
1. Install dependencies: pip install -U openai neo4j qdrant-client llama-index pypdf pytesseract Pillow openai-whisper opencv-python streamlit deepeval
2. Install Docker and Neo4J Desktop
3. In Neo4J, create a new graph DB
4. Install qdrant using Docker in command line:
docker run -d \
  --name qdrant-rag \
  -p 6333:6333 \
  -p 6334:6334 \    # optional gRPC
  qdrant/qdrant:latest
5. This app uses Streamlit for UI. Run app using: streamlit run aparavi.py

Demo Video:
