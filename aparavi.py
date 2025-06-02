# Imports
import os
import uuid
import logging
import traceback
from pypdf import PdfReader
import pytesseract
from PIL import Image
import whisper
from openai import OpenAI
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import streamlit as st
import pandas as pd


# Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Configure LlamaIndex global settings
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
Settings.llm = LlamaOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

logging.basicConfig(filename='rag.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Evaluation suite (using deepeval)
def test_suite():
    test_cases = [
    LLMTestCase(
      input="What is the capital of France?",
      expected_output="Paris",
      metrics=[GEval(name="Accuracy", criteria="Is the response factually correct?")]
    ),
    LLMTestCase(
      input="Summarize document X.",
      expected_output="Summary of document X.",
      metrics=[GEval(name="Summarization Quality", criteria="Does the summary capture key points?")]
    ),
    LLMTestCase(
      input="How is John related to Intel Corp?",
      expected_output="John is the CEO of Intel Corp.",
      metrics=[GEval(name="Relationship Accuracy", criteria="Is the relationship correctly identified?")]
    ),
    LLMTestCase(
      input="What is the meaning of life?",
      expected_output="No information found",
      metrics=[GEval(name="Hallucination Control", criteria="Does the system avoid making up information?")]
    ),
    LLMTestCase(
      input="Tell me about John Smith.",
      expected_output="John Smith is mentioned in document A and transcript B.",
      metrics=[GEval(name="Cross-Modal Linking", criteria="Does the system link information across modalities?")]
    )
    ]
    return test_cases


def evaluate_query_response(query, answer, context):
  """Evaluate a query response and log the results"""
  try:
    # Create test case for this query
    test_case = LLMTestCase(
      input=query,
      actual_output=answer
    )
    
    # Define metrics with proper evaluation_params
    metrics = [
      GEval(
        name="Relevance", 
        criteria="Is the response relevant to the query?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
      ),
      GEval(
        name="Accuracy", 
        criteria="Is the response factually accurate based on available context?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
      ),
      GEval(
        name="Completeness", 
        criteria="Does the response adequately address the query?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
      ),
      GEval(
        name="Hallucination Control", 
        criteria="Does the system avoid making up information not in the context?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
      )
    ]
    # Evaluate the test case
    for metric in metrics:
      metric.measure(test_case)
      score = metric.score
      logging.info(f"Query Evaluation - {metric.name}: {score:.3f} | Query: {query[:50]}... | Answer: {answer[:100]}...")
      
    # Create a simple test case object to return
    class EvaluationResult:
      def __init__(self):
        self.input = query
        self.actual_output = answer
        self.metrics = metrics
    
    return EvaluationResult()
  except Exception as e:
    logging.error(f"Error evaluating query response: {e}")
    return None


# Validate input
def validate_input(path):
  extensions = (".txt", ".pdf", ".mp3", ".jpg", ".png")
  if not path.lower().endswith(extensions):
    logging.error(f"Unsupported file type: {path}")
    return False
  return True


# Data ingestion
def ingest_file(file_path):
  if not validate_input(file_path):
    raise ValueError("Invalid file")
  try:
    if file_path.endswith('.txt'):
      with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
      metadata = {'file_name': os.path.basename(file_path), 'type': 'text', 'id': str(uuid.uuid4()), 'domain':'general'}
    elif file_path.endswith('.pdf'):
      with open(file_path, 'rb') as f:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
          text += page.extract_text()
      metadata = {'file_name': os.path.basename(file_path), 'type': 'pdf', 'id': str(uuid.uuid4()), 'domain':'general'}
    elif file_path.endswith('.mp3'):
      model = whisper.load_model("base")
      result = model.transcribe(file_path)
      text = result["text"]
      metadata = {'file_name': os.path.basename(file_path), 'type': 'audio', 'id': str(uuid.uuid4()), 'domain':'general'}
    elif file_path.endswith(('.jpg', '.png')):
      image = Image.open(file_path)
      text = pytesseract.image_to_string(image) # OCR
      metadata = {'file_name': os.path.basename(file_path), 'type': 'image', 'id': str(uuid.uuid4()), 'domain':'general'}
    else:
      raise ValueError("Unsupported file type")
    return text, metadata
  except Exception as e:
    logging.error(f"Error extracting text from {file_path}: {e}")
    raise


# Extract entities and relationships using LLM
def extract_entities_relationships(text):
  prompt = f"""
  Extract entities and relationships from the text. Output format:
  Entities:
  - Entity: type
  Relationships:
  - Entity1 - Relationship - Entity2
  Text: {text}
  """

  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": prompt}]
  )
  output = response.choices[0].message.content.strip()
  entities = {}
  relationships = []
  current_section = None

  for line in output.split('\n'):
    line = line.strip()
    if line.startswith("Entities"):
      current_section = "Entities"
    elif line.startswith("Relationships"):
      current_section = "Relationships"
    elif current_section == "Entities" and line.startswith("- "):
      if ": " in line[2:]:
        entity, type_ = line[2:].split(": ", 1)
        entities[entity] = type_
    elif current_section == "Relationships" and line.startswith("- "):
      parts = line[2:].split(" - ")
      if len(parts) == 3:
        e1, rel, e2 = parts
        relationships.append((e1, rel, e2))

  return entities, relationships


# Knowledge graph (using Neo4J)
def construct_knowledge_graph(entities, relationships):
  with neo4j_driver.session() as session:
    for entity, type_ in entities.items():
      session.run(
          "MERGE (e:Entity {name: $name}) SET e.type = $type",
          name=entity, type=type_
      )
    for e1, rel, e2 in relationships:
      session.run(
          """
          MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2})
          MERGE (e1)-[r:RELATION {type: $rel}]->(e2)
          """,
          entity1=e1, entity2=e2, rel=rel
      )


# Process query
def process_query(query):
  prompt = f"""Classify this query into exactly one of these types:
    - lookup: For finding specific information, facts, or details about people, companies, projects, or topics from documents and knowledge base
    - summarization: For requesting summaries, overviews, or consolidated information from multiple sources
    - semantic_linkage: ONLY for very simple relationship questions like "Who works for Company X?" or "What companies does Person Y work for?"

    Guidelines:
    - Most "how is X related to Y" questions should be "lookup" since they often require document context
    - Only use "semantic_linkage" for simple direct relationship queries

    Query: {query}

    Respond with only the type name, nothing else."""
  
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": prompt}]
  )
  output = response.choices[0].message.content.strip().lower()
  
  if "lookup" in output:
    qtype = "lookup"
  elif "summarization" in output:
    qtype = "summarization"
  elif "semantic_linkage" in output:
    qtype = "semantic_linkage"
  else:
    qtype = "lookup" # default
  
  return qtype, query


# Retrieval agent (orchestration)
class RetrievalAgent:
  def __init__(self, index, graph_driver):
    self.index = index
    self.graph_driver = graph_driver
    self.vector_retriever = VectorIndexRetriever(index=index)
    # Only create BM25 retriever if index has documents
    self.bm25_retriever = self._create_bm25_retriever()

  def _create_bm25_retriever(self):
    try:
      if len(self.index.docstore.docs) > 0:
        return BM25Retriever.from_defaults(index=self.index)
      else:
        return None
    except:
      return None
  
  def refresh_retrievers(self):
    """Call this after adding new documents to reinitialize retrievers"""
    self.vector_retriever = VectorIndexRetriever(index=self.index)
    self.bm25_retriever = self._create_bm25_retriever()

  def graph_search(self, entity):
    with self.graph_driver.session() as session:
      result = session.run(
          "MATCH (e1:Entity {name: $name})-[r]->(e2:Entity) "
          "RETURN e1.name, type(r), e2.name LIMIT 5",
          name=entity
      )
      relationships = [(record["e1.name"], record["type(r)"], record["e2.name"]) for record in result]
      
      # Also try reverse direction and partial matches
      if not relationships:
        result = session.run(
            "MATCH (e1:Entity)-[r]->(e2:Entity {name: $name}) "
            "RETURN e1.name, type(r), e2.name LIMIT 5",
            name=entity
        )
        relationships = [(record["e1.name"], record["type(r)"], record["e2.name"]) for record in result]
      
      # Try partial name matching if exact match fails
      if not relationships:
        result = session.run(
            "MATCH (e1:Entity)-[r]->(e2:Entity) "
            "WHERE e1.name CONTAINS $name OR e2.name CONTAINS $name "
            "RETURN e1.name, type(r), e2.name LIMIT 5",
            name=entity
        )
        relationships = [(record["e1.name"], record["type(r)"], record["e2.name"]) for record in result]
      
      return relationships

  def retrieve(self, qtype, query):
    try:
      if qtype == "lookup":
        vector_nodes = self.vector_retriever.retrieve(query)[:3] if len(self.index.docstore.docs) > 0 else []
        if self.bm25_retriever:
          bm25_nodes = self.bm25_retriever.retrieve(query)[:3]
        else:
          bm25_nodes = []
        
        # Extract potential entity from query
        entities_to_try = []
        query_words = query.replace("?", "").split()
        
        # Try proper nouns and common entity patterns
        for word in query_words:
          if word[0].isupper() and len(word) > 2:
            entities_to_try.append(word)
        
        # Try multi-word entities
        for i in range(len(query_words)):
          for j in range(i+1, min(i+4, len(query_words)+1)):
            phrase = " ".join(query_words[i:j])
            if any(w[0].isupper() for w in phrase.split()):
              entities_to_try.append(phrase)
        
        graph_results = []
        for entity in entities_to_try[:3]:  # Try top 3 entities
          graph = self.graph_search(entity)
          if graph:
            graph_results.extend(graph)
            break  # Use first successful match
        
        texts = [node.text for node in vector_nodes] + [node.text for node in bm25_nodes] + [f"{e1} - {rel} - {e2}" for e1, rel, e2 in graph_results[:3]]
        unique_texts = []
        seen = set()
        for text in texts:
          if text not in seen:
            unique_texts.append(text)
            seen.add(text)
        return unique_texts[:5]
      elif qtype == "summarization":
        vector_nodes = self.vector_retriever.retrieve(query)[:5]
        return [node.text for node in vector_nodes]
      elif qtype == "semantic_linkage":
        # Extract entities from query
        entities_to_try = []
        query_words = query.replace("?", "").split()
        
        for word in query_words:
          if word[0].isupper() and len(word) > 2:
            entities_to_try.append(word)
        
        for i in range(len(query_words)):
          for j in range(i+1, min(i+4, len(query_words)+1)):
            phrase = " ".join(query_words[i:j])
            if any(w[0].isupper() for w in phrase.split()):
              entities_to_try.append(phrase)
        
        graph_results = []
        for entity in entities_to_try:
          try:
            graph = self.graph_search(entity)
            graph_results.extend(graph)
          except Exception as e:
            logging.error(f"Error searching for '{entity}': {e}")
        
        return [f"{e1} - {rel} - {e2}" for e1, rel, e2 in graph_results[:5]]
      else:
        return ["Query type not recognized"]
    except Exception as e:
      logging.error(f"Error in retrieval: {e}")
      return [f"Retrieval failed: {str(e)}"]



def generate_answer(llm, query, context):
  if not context or context == ["Retrieval failed"] or context == ["Query type not recognized"]:
    return "No information found in the knowledge base."
  
  formatted_context = []
  for item in context:
    if " - RELATION - " in item:
      parts = item.split(" - RELATION - ")
      if len(parts) == 2:
        entity1, entity2 = parts
        formatted_context.append(f"{entity1} is related to {entity2}")
      else:
        formatted_context.append(item)
    else:
      formatted_context.append(item)
  
  context_input = "\n".join(formatted_context)
  
  prompt = f"""Based on the following context from a knowledge base, answer the user's question. The context includes relationships between entities and document content.
  Question: {query}
  Context: {context_input}
  
  Instructions:
  - If you find relevant relationships or information, provide a clear answer
  - For relationship queries, explain how entities are connected
  - If the context shows connections but lacks detail, acknowledge what you can determine
  - Only say "No relevant information found" if the context is truly unrelated to the question
  Answer:"""
  
  try:
    response = llm.complete(prompt)
    return response.text.strip()
  except Exception as e:
    logging.error(f"Error generating answer: {e}")
    return f"Error generating answer: {str(e)}"



def main():
  st.title("Enterprise RAG System")
  st.write("🔧 **System Status:**")
  
  try:
    qdrant_client.get_collections()
    st.success("✅ Qdrant: Connected")
    qdrant_ok = True
  except Exception as e:
    st.error(f"❌ Qdrant: Failed - {e}")
    qdrant_ok = False
  
  try:
    with neo4j_driver.session() as session:
      session.run("RETURN 1")
    st.success("✅ Neo4j: Connected")
    neo4j_ok = True
  except Exception as e:
    st.error(f"❌ Neo4j: Failed - {e}")
    neo4j_ok = False
  
  try:
    client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": "test"}],
      max_tokens=1
    )
    st.success("✅ OpenAI: Connected")
    openai_ok = True
  except Exception as e:
    st.error(f"❌ OpenAI: Failed - {e}")
    openai_ok = False
  
  if not all([qdrant_ok, openai_ok]):
    st.error("⚠️ Some services are unavailable. Please check connections.")
    return
  
  try:
    st.write("🔧 Initializing vector store...")
    collection_name = "enterprise_rag"
    
    # Create the vector store
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    st.write("🔧 Creating index...")
    try:
      index = VectorStoreIndex.from_vector_store(vector_store, storage_context)
      st.write("✅ Loaded existing index")
    except Exception as e:
      st.write(f"⚠️ Creating new index: {e}")
      embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
      index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
      st.write("✅ Created new index with explicit embedding model")
    
    st.write("🔧 Initializing LLM...")
    llm = LlamaOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    
    if neo4j_ok:
      st.write("🔧 Initializing retrieval agent...")
      retrieval_agent = RetrievalAgent(index, neo4j_driver)
    else:
      st.warning("⚠️ Running without Neo4j - graph features disabled")
      retrieval_agent = None
      
    st.success("🚀 System initialized successfully!")
    
  except Exception as e:
    st.error(f"❌ Initialization failed: {e}")
    st.write(f"Error details: {str(e)}")
    st.code(traceback.format_exc())
    return

  uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=['txt', 'pdf', 'jpg', 'png', 'mp3'])

  if uploaded_files:
    for file in uploaded_files:
      file_path = os.path.join("uploads", file.name)
      with open(file_path, "wb") as f:
        f.write(file.getbuffer())
      try:
        st.write(f"🔍 Processing {file.name}...")
        text, metadata = ingest_file(file_path)
        st.write(f"📄 Extracted {len(text)} characters")
        
        if neo4j_ok:
          st.write("🔍 Extracting entities and relationships...")
          entities, relationships = extract_entities_relationships(text)
          st.write(f"Found {len(entities)} entities, {len(relationships)} relationships")
          construct_knowledge_graph(entities, relationships)
        
        st.write("🔍 Adding to vector index...")
        document = Document(text=text, metadata=metadata)
        
        # Generate embedding for document
        embed_model = Settings.embed_model
        if embed_model:
          embedding = embed_model.get_text_embedding(text[:1000])
          document.embedding = embedding
        
        # Add document to index
        try:
          index.docstore.add_documents([document])
          all_docs = list(index.docstore.docs.values())
          index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
          
          final_count = len(index.docstore.docs)
          st.write(f"📊 Final vector index count: {final_count} documents")
          
        except Exception as e:
          st.write(f"❌ Indexing failed: {e}")
          st.code(traceback.format_exc())
        
        # Refresh retrievers if we have a retrieval agent
        if retrieval_agent:
          retrieval_agent.refresh_retrievers()
        
        st.success(f"Ingested {file.name}")
      except Exception as e:
        st.error(f"Error ingesting {file.name}: {e}")
        st.code(traceback.format_exc())

  query = st.text_input("Enter your query")
  if query:
    try:
      st.write("🔍 **Processing Query...**")
      query_type, rewritten_query = process_query(query)
      st.write(f"Query Type: {query_type}")
      st.write(f"Rewritten Query: {rewritten_query}")
      
      if retrieval_agent:
        st.write("🔍 **Retrieving Context...**")
        context = retrieval_agent.retrieve(query_type, rewritten_query)
        st.write(f"Retrieved {len(context)} context items")
      else:
        st.write("🔍 **Using Simple Vector Search...**")
        from llama_index.core.retrievers import VectorIndexRetriever
        simple_retriever = VectorIndexRetriever(index=index)
        nodes = simple_retriever.retrieve(query)
        context = [node.text for node in nodes[:5]]
        st.write(f"Retrieved {len(context)} context items from vector search")
      
      if context and context != ["Query type not recognized"] and context != ["Retrieval failed"]:
        answer = generate_answer(llm, rewritten_query, context)
      else:
        answer = "No relevant information found in the knowledge base."
      
      st.write(f"**Answer:** {answer}")
      
      logging.info(f"Query: {query}\nAnswer: {answer}")
      
      # Evaluate the query response
      test_case = evaluate_query_response(query, answer, context)
      if test_case:
        with st.expander("📊 Query Evaluation Scores"):
          for metric in test_case.metrics:
            st.write(f"**{metric.name}:** {metric.score:.3f}")
    except Exception as e:
      st.error(f"Error processing query: {e}")
      st.code(traceback.format_exc())
      logging.error(f"Error processing query: {e}")


if __name__ == "__main__":
  if not os.path.exists("uploads"):
    os.mkdir("uploads")
  main()