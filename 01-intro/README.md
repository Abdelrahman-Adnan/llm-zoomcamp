# 🚀 LLM Zoomcamp Week 1: Retrieval-Augmented Generation (RAG)

## 🧩 The Big Picture: What is RAG?

Retrieval-Augmented Generation (RAG) is like giving your AI assistant a personalized reference library! It combines:

1. **Retrieval** 🔍: A smart search engine that finds relevant facts
2. **Generation** ✍️: An LLM that crafts human-like responses

## 🤖 Why Do We Need RAG?

Standard LLMs have two major limitations:
- They only know what they were trained on (often outdated) ⏰
- They sometimes hallucinate (make things up) 🤪

RAG solves both problems by feeding the model fresh, relevant information at query time!

## 🛠️ How RAG Actually Works

### The Simple Flow:

```
User Question → Search Engine → Relevant Docs → Enhanced Prompt → LLM → Answer
```

### The Magic Principle 💫

LLMs prioritize information in the immediate context over what they've memorized during training. By placing facts directly in the prompt, we dramatically increase accuracy!

## 🔧 Building Your Own RAG System

### 1️⃣ Set Up Your Knowledge Base

```python
# Download your documents (FAQs, manuals, etc.)
!wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json

# Start ElasticSearch (lightweight search engine)
!docker run -d -p 9200:9200 -p 9300:9300 -m 4g \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

### 2️⃣ Index Your Documents

```python
# Create an index with proper text mappings
es.indices.create(index="course_questions", settings={"number_of_shards": 1, "number_of_replicas": 0})

# Add all documents to ElasticSearch
for doc in documents:
    es.index(index="course_questions", document=doc)
```

### 3️⃣ Create Your Search Function

```python
def elastic_search(query: str) -> list:
    """Find documents relevant to the query"""
    search_query = {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {"term": {"course": "data-engineering-zoomcamp"}}
            }
        },
        "size": 5
    }
    
    response = es.search(index="course_questions", body=search_query)
    results = [hit["_source"] for hit in response["hits"]["hits"]]
    return results
```

### 4️⃣ Build Your Prompt Constructor

```python
def build_prompt(query: str, search_results: list) -> str:
    """Format retrieved context and query into effective prompt"""
    
    # Format context from search results
    context_parts = []
    for i, doc in enumerate(search_results, 1):
        context_parts.append(
            f"DOCUMENT {i}:\n"
            f"Section: {doc['section']}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['text']}\n"
        )
    context = "\n".join(context_parts)
    
    # Create instruction-focused prompt
    prompt = f"""You are a course teaching assistant for the DataTalks.Club courses.
CONTEXT:
{context}

QUESTION: {query}

Answer the QUESTION using ONLY information from the CONTEXT.
If you cannot answer based on the CONTEXT alone, say "I don't have enough information to answer that."
Keep your answer concise and helpful for a student."""
    
    return prompt.strip()
```

### 5️⃣ Connect to an LLM

```python
def llm(prompt: str, model: str = "llama3-8b-8192") -> str:
    """Generate response using LLM API"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return completion.choices[0].message.content
```

### 6️⃣ Combine Everything in a RAG Pipeline

```python
def rag(query: str, model: str = "llama3-8b-8192") -> str:
    """Complete RAG pipeline from query to answer"""
    # 1. Retrieve relevant documents
    search_results = elastic_search(query)
    
    # 2. Build enhanced prompt with context
    prompt = build_prompt(query, search_results)
    
    # 3. Generate answer with LLM
    answer = llm(prompt, model)
    
    return answer
```

## 🔥 Real-World Examples

### Example 1: Course FAQ

**User**: "How do I run Kafka locally?"

**Without RAG**: *"To run Kafka locally, you typically need to download Kafka, extract it, start Zookeeper, then start the Kafka server. You can do this by running bin/zookeeper-server-start.sh and bin/kafka-server-start.sh with their respective configuration files..."* (Potentially incorrect or outdated)

**With RAG**: *"According to the course materials, you can run Kafka locally using Docker Compose. The docker-compose.yml file is provided in the 'week_6_stream_processing' folder. Simply navigate to that directory and run 'docker-compose up' to start Kafka and Zookeeper together."* (Accurate based on course-specific instructions)

### Example 2: Technical Troubleshooting

**User**: "My Spark job is failing with OOM errors"

**Without RAG**: *"Out of Memory errors in Spark can happen for various reasons..."* (Generic advice)

**With RAG**: *"In the Data Engineering Zoomcamp, this issue is addressed in the FAQ. You should try increasing the memory allocation for the Spark driver and executor by setting 'spark.driver.memory' and 'spark.executor.memory' in your SparkSession configuration. The course specifically recommends starting with '2g' for both settings when working with the NYC taxi dataset."* (Specific to course materials)

## 📈 Advanced RAG Techniques

### 🧠 1. Contextual Compression
Only include the most relevant parts of documents to save context window space.

### 🔄 2. Hybrid Search
Combine keyword search with embedding-based semantic search for better results.

### 🧪 3. Self-Reflection
Have the LLM check if its answer actually addresses the question before responding.

## 🌟 When to Use RAG

RAG works best for:
- Question-answering over specific documents 📚
- Customer support with product documentation 🛎️
- Technical assistance with code/API docs 👨‍💻
- Domain-specific knowledge (medical, legal, etc.) 👩‍⚕️

## 🚦 Getting Started Checklist

1. ✅ Gather your knowledge documents
2. ✅ Set up a search system (ElasticSearch/Pinecone/Weaviate)
3. ✅ Connect to an LLM API (OpenAI/Groq/Anthropic)
4. ✅ Create your prompt template
5. ✅ Build the retrieval-generation pipeline
6. ✅ Test with real questions!

## 🎬 Final Thoughts

RAG isn't just a technical solution—it's a paradigm shift in how we think about AI assistants. Instead of expecting models to "know everything," we give them the specific knowledge they need, when they need it! 🧠✨

Remember: The quality of your RAG system depends on the quality of your documents, your search capability, and your prompt engineering. Keep improving all three for the best results!