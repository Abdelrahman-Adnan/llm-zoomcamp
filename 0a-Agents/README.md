# 🤖 Building Agentic Assistants with OpenAI Function Calling: A Complete LLM Zoomcamp Tutorial

Welcome to the **LLM Zoomcamp** comprehensive tutorial! 🎓 This guide explores how to build smart, agentic assistants using OpenAI Function Calling, drawing from advanced workshop materials. We'll journey from basic RAG (Retrieval Augmented Generation) to sophisticated agentic systems that can make autonomous decisions and use tools.

Perfect for **LLM Zoomcamp** students who want to master the art of building intelligent AI assistants! 🚀

## 📚 Table of Contents
- [🏗️ Part 1: Foundation - Basic RAG and Agentic Concepts](#️-part-1-foundation---basic-rag-and-agentic-concepts)
- [⚡ Part 2: Advanced Implementation - Function Calling and Tool Integration](#-part-2-advanced-implementation---function-calling-and-tool-integration)
- [🏭 Part 3: Production Ready - Object-Oriented Design and Libraries](#-part-3-production-ready---object-oriented-design-and-libraries)

---

## 🏗️ Part 1: Foundation - Basic RAG and Agentic Concepts

### 🎯 Understanding the Core Problem (LLM Zoomcamp Challenge)

Welcome to your first **LLM Zoomcamp** agentic project! 🎓 Our goal is to create an **intelligent assistant** that can help course participants by leveraging **Frequently Asked Questions (FAQ) documents**. These FAQ documents contain question-answer pairs that provide valuable information about course enrollment, requirements, and procedures.

Think of it like having a smart study buddy who has read all the course materials! 📖

#### 🎯 What We Want to Build:
- 🔍 Search through FAQ documents intelligently
- 🧠 Decide when to use external knowledge vs. built-in knowledge
- 🔄 Make multiple search iterations for complex queries
- 💬 Provide contextual, accurate responses

### 🤖 What Makes a System "Agentic"? (LLM Zoomcamp Core Concept)

An **agent** in AI is like a smart assistant that can think and act independently! Here's what makes it special:

🌍 **Interacts with an environment** (in our case, the chat dialogue)  
👀 **Observes and gathers information** (through search functions)  
🏃‍♂️ **Performs actions** (searching, answering, adding entries)  
🧠 **Maintains memory** of past actions and context  
⚡ **Makes independent decisions** about what to do next  

The key difference between basic RAG and agentic RAG is **decision-making autonomy**! Instead of always searching or always using built-in knowledge, an agentic system can intelligently choose the best approach. 🎯

### 🏗️ Building Basic RAG Foundation (LLM Zoomcamp Step-by-Step)

Let's start by building the fundamental building blocks! Think of this as learning to walk before we run. 👶

#### 🛠️ Step 1: Setting Up Your LLM Zoomcamp Environment

```python
# 📦 First, let's install the packages we need
# Think of these as your toolkit for building AI assistants!
pip install openai minsearch requests jupyter markdown
```

Now let's import our tools one by one:

```python
# 📚 Import the libraries (like getting books from a library)
import json          # For working with data in JSON format
import requests      # For downloading data from the internet
from openai import OpenAI         # For talking to ChatGPT
from minsearch import AppendableIndex  # For searching through documents

# 🔑 Initialize OpenAI client (this is your key to ChatGPT)
# Make sure you have OPENAI_API_KEY set in your environment!
client = OpenAI()
```

**🎓 LLM Zoomcamp Tip**: Think of the OpenAI client as your telephone to ChatGPT. You'll use it to send questions and get answers!

#### 📊 Step 2: Getting and Preparing Our LLM Zoomcamp Data

Now let's get some real FAQ data to work with! This is like downloading all the course materials. 📥

```python
# 🌐 Step 2a: Download the FAQ documents from the internet
# This URL contains real FAQ data from data engineering courses
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

print("📥 Downloaded FAQ data successfully!")
print(f"📊 Found {len(documents_raw)} courses with FAQ data")
```

**🎓 LLM Zoomcamp Explanation**: We're downloading a JSON file that contains FAQ questions and answers from real courses. Think of it as a digital textbook! 📚

```python
# 🔄 Step 2b: Transform the data into a format we can search
# We're "flattening" the data - turning nested data into a simple list
documents = []

for course in documents_raw:  # Go through each course
    course_name = course['course']  # Get the course name
    
    for doc in course['documents']:  # Go through each FAQ in that course
        doc['course'] = course_name  # Add course name to each FAQ
        documents.append(doc)        # Add to our main list

print(f"✅ Processed {len(documents)} FAQ documents total!")
print("📋 Each document now has: question, answer, section, and course name")
```

**🎓 LLM Zoomcamp Explanation**: Imagine you have several books (courses), each with many pages (documents). We're taking all the pages and putting them in one big stack, but we label each page with which book it came from! 📚➡️📄

```python
# 🗂️ Step 2c: Create our search index (like a super-smart filing cabinet)
index = AppendableIndex(
    text_fields=["question", "text", "section"],  # Fields we can search in
    keyword_fields=["course"]                     # Fields for exact filtering
)

# 🚀 Put all our documents into the search index
index.fit(documents)

print("🗂️ Created search index successfully!")
print("🔍 Now we can quickly find relevant FAQ answers!")
```

**🎓 LLM Zoomcamp Explanation**: Think of this index like Google for your FAQ documents. Instead of reading every single document, we can ask "find me documents about Docker" and it will instantly find the relevant ones! ⚡

#### 🔍 Step 3: Building Our LLM Zoomcamp Search Function

Now let's create a function that can search through our FAQ documents! This is like having a research assistant. 🕵️‍♀️

```python
def search(query):
    """
    🔍 Search the FAQ database for relevant entries.
    
    Think of this as asking a librarian: "Can you find me books about Python?"
    
    Args:
        query (str): What the user wants to search for (like "Docker setup")
        
    Returns:
        list: A list of relevant FAQ entries, ranked by relevance
    """
    
    # 🎯 Step 3a: Set up boosting (some fields are more important)
    # Questions are 3x more important than sections when matching
    boost = {
        'question': 3.0,    # If the search term appears in a question, it's very relevant!
        'section': 0.5      # If it appears in a section name, it's somewhat relevant
    }
    
    # 🔍 Step 3b: Actually perform the search
    results = index.search(
        query=query,                                    # What to search for
        filter_dict={'course': 'data-engineering-zoomcamp'},  # Only search in this course
        boost_dict=boost,                               # Use our importance scoring
        num_results=5,                                  # Return top 5 matches
        output_ids=True                                 # Include document IDs
    )
    
    return results

# 🧪 Let's test our search function!
test_results = search("How do I install Docker?")
print(f"🔍 Found {len(test_results)} results for 'How do I install Docker?'")

# 👀 Let's look at the first result
if test_results:
    first_result = test_results[0]
    print(f"📝 First result question: {first_result['question']}")
    print(f"⭐ Relevance score: {first_result.get('score', 'N/A')}")
```

**🎓 LLM Zoomcamp Explanation**: Our search function is like a smart librarian who:
1. 🎯 Knows that questions are more important than section names
2. 🔍 Only looks in the specific course we care about
3. ⭐ Ranks results by how well they match
4. 📊 Returns the top 5 most relevant answers

#### 🏗️ Step 4: Creating Our LLM Zoomcamp RAG Pipeline

Now we'll build the complete RAG system step by step! RAG = **R**etrieval + **A**ugmented + **G**eneration. 🏗️

```python
# 📝 Step 4a: Helper function to format search results
def build_context(search_results):
    """
    🏗️ Build a context string from search results.
    
    Think of this as organizing your research notes before writing an essay!
    
    Args:
        search_results (list): Results from our search function
        
    Returns:
        str: Nicely formatted context for the AI to use
    """
    context = ""
    
    # 🔄 Go through each search result and format it nicely
    for doc in search_results:
        context += f"section: {doc['section']}\n"          # What section this is from
        context += f"question: {doc['question']}\n"        # The original question
        context += f"answer: {doc['text']}\n\n"           # The answer text
    
    return context.strip()  # Remove extra whitespace

# 🧪 Let's test our context builder
test_results = search("Docker installation")
test_context = build_context(test_results)
print("📝 Built context from search results:")
print(test_context[:200] + "..." if len(test_context) > 200 else test_context)
```

**🎓 LLM Zoomcamp Explanation**: The `build_context` function is like organizing your research notes. Instead of giving ChatGPT a messy pile of information, we organize it neatly so the AI can easily understand and use it! 📋

```python
# 🤖 Step 4b: Function to talk to ChatGPT
def llm(prompt):
    """
    🤖 Send a question to ChatGPT and get an answer back.
    
    This is like having a conversation with a very smart assistant!
    
    Args:
        prompt (str): The complete question/instruction for ChatGPT
        
    Returns:
        str: ChatGPT's response
    """
    
    # 📞 Make the API call to OpenAI
    response = client.chat.completions.create(
        model='gpt-4o-mini',                           # Which AI model to use
        messages=[{"role": "user", "content": prompt}] # Our question
    )
    
    # 📥 Extract the text response
    return response.choices[0].message.content

print("🤖 LLM function ready - we can now talk to ChatGPT!")
```

**🎓 LLM Zoomcamp Explanation**: This function is your hotline to ChatGPT! You send it a prompt (like a detailed question), and it sends back ChatGPT's answer. Simple! 📞➡️🤖➡️💬

```python
# 🎯 Step 4c: Create the main RAG function
def basic_rag(query):
    """
    🎯 Our complete RAG pipeline: Search + Context + Generate Answer
    
    This is the magic! We combine search results with AI to answer questions.
    
    Args:
        query (str): The user's question (like "How do I join the course?")
        
    Returns:
        str: A complete, helpful answer
    """
    
    # 🔍 Step 1: Search for relevant information
    print(f"🔍 Searching for: {query}")
    search_results = search(query)
    
    # 📝 Step 2: Build context from search results
    print(f"📝 Found {len(search_results)} relevant documents")
    context = build_context(search_results)
    
    # 🎭 Step 3: Create a detailed prompt for ChatGPT
    prompt_template = """
You're a helpful course teaching assistant for the LLM Zoomcamp! 🎓

Your job is to answer the QUESTION based on the CONTEXT from our FAQ database.
Only use facts from the CONTEXT when answering the QUESTION.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>

Please provide a helpful, detailed answer! 😊
""".strip()
    
    # 🔄 Step 4: Fill in the template with our data
    prompt = prompt_template.format(question=query, context=context)
    print("🎭 Created prompt for ChatGPT")
    
    # 🤖 Step 5: Get answer from ChatGPT
    print("🤖 Getting answer from ChatGPT...")
    answer = llm(prompt)
    
    return answer

# 🧪 Let's test our complete RAG system!
print("🧪 Testing our LLM Zoomcamp RAG system!")
test_question = "How do I join the course?"
answer = basic_rag(test_question)

print(f"\n❓ Question: {test_question}")
print(f"✅ Answer: {answer}")
```

**🎓 LLM Zoomcamp Explanation**: Our `basic_rag` function is like having a research assistant who:
1. 🔍 Searches through all course materials
2. 📝 Organizes the relevant information
3. 🎭 Asks ChatGPT a well-structured question
4. ✅ Returns a helpful answer based on real course data!

### 🧠 Making RAG Agentic: Decision-Making Capabilities (LLM Zoomcamp Advanced)

The basic RAG always searches first, then answers. But what if we want our system to be smarter? 🤔 An agentic system should decide whether to search or use its own knowledge. Let's make it intelligent! 🧠

#### 🎭 Enhanced Agentic Prompt (LLM Zoomcamp Magic)

```python
# 🎭 This is our "smart prompt" that teaches ChatGPT to make decisions
agentic_prompt_template = """
🎓 You're a course teaching assistant for the LLM Zoomcamp!

You're given a QUESTION from a student. You have three superpowers:

1. 📖 Answer using the provided CONTEXT (if available and good enough)
2. 🧠 Use your own knowledge if CONTEXT is EMPTY or not helpful  
3. 🔍 Request a search of the FAQ database if you need more info

Current CONTEXT: {context}

<QUESTION>
{question}
</QUESTION>

🔍 If CONTEXT is EMPTY or you need more information, respond with:
{{
"action": "SEARCH",
"reasoning": "Explain why you need to search the FAQ database"
}}

📖 If you can answer using CONTEXT, respond with:
{{
"action": "ANSWER",
"answer": "Your detailed, helpful answer here",
"source": "CONTEXT"
}}

🧠 If CONTEXT isn't helpful but you can answer from your knowledge:
{{
"action": "ANSWER", 
"answer": "Your detailed, helpful answer here",
"source": "OWN_KNOWLEDGE"
}}

Remember: Always be helpful and explain things clearly! 😊
""".strip()

print("🎭 Created our intelligent agentic prompt!")
print("✨ Now ChatGPT can decide what to do instead of always searching!")
```

**🎓 LLM Zoomcamp Explanation**: This prompt is like giving ChatGPT a decision-making flowchart! Instead of always doing the same thing, it can now choose the best action based on the situation. It's like upgrading from a calculator to a smartphone! 📱

#### 🚀 Implementing Agentic Decision Logic (LLM Zoomcamp Step-by-Step)

Now let's build our smart assistant that can make decisions! This is where the magic happens! ✨

```python
def agentic_rag_v1(question):
    """
    🚀 First version of our smart agentic RAG system!
    
    This assistant can decide whether to search or use its own knowledge.
    
    Args:
        question (str): The student's question
        
    Returns:
        dict: The assistant's response with source information
    """
    
    # 🎬 Step 1: Start with empty context (no information yet)
    print(f"🎬 Starting with question: {question}")
    context = "EMPTY"
    
    # 🎭 Step 2: Create prompt and ask ChatGPT what to do
    prompt = agentic_prompt_template.format(question=question, context=context)
    print("🤔 Asking ChatGPT to make a decision...")
    
    # 🤖 Step 3: Get ChatGPT's decision
    answer_json = llm(prompt)
    answer = json.loads(answer_json)  # Convert JSON string to Python dictionary
    
    print(f"🧠 ChatGPT decided: {answer['action']}")
    
    # 🔍 Step 4: If ChatGPT wants to search, let's do it!
    if answer['action'] == 'SEARCH':
        print(f"🔍 Reason for searching: {answer['reasoning']}")
        print("📚 Performing search...")
        
        # Search the FAQ database
        search_results = search(question)
        context = build_context(search_results)
        
        print(f"✅ Found {len(search_results)} relevant documents")
        
        # Ask ChatGPT again, now with context
        prompt = agentic_prompt_template.format(question=question, context=context)
        print("🤖 Asking ChatGPT again with search results...")
        
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        
        print(f"✨ Final decision: {answer['action']}")
    
    return answer

# 🧪 Let's test our smart assistant!
print("🧪 Testing LLM Zoomcamp Agentic Assistant!")
print("\n" + "="*50)

# Test 1: Course-specific question (should search)
print("📚 Test 1: Course-specific question")
result1 = agentic_rag_v1("How do I join the LLM Zoomcamp course?")
print(f"📝 Answer: {result1['answer'][:200]}...")
print(f"🏷️ Source: {result1['source']}")

print("\n" + "="*50)

# Test 2: General knowledge question (should use own knowledge)
print("🌍 Test 2: General knowledge question")
result2 = agentic_rag_v1("How do I install Python on my computer?")
print(f"📝 Answer: {result2['answer'][:200]}...")
print(f"🏷️ Source: {result2['source']}")
```

**🎓 LLM Zoomcamp Explanation**: Our smart assistant works like this:

1. 🤔 **Think First**: "Do I need to search, or do I already know this?"
2. 🔍 **Search If Needed**: If it's about the course, search the FAQ
3. 🧠 **Use Knowledge**: If it's general knowledge, answer directly
4. 📝 **Always Cite Sources**: Tell us where the answer came from!

It's like having a study buddy who knows when to check the textbook vs. when they already know the answer! 🎓

### 🎓 Key Concepts Introduced in Part 1 (LLM Zoomcamp Fundamentals)

Congratulations! You've just built your first intelligent agent! 🎉 Let's review what you've learned:

1. **🏗️ RAG Pipeline**: Search → Context Building → LLM Query
   - Like having a research assistant who finds info, organizes it, and writes an answer

2. **🧠 Agentic Decision Making**: LLM chooses actions based on available information
   - Your assistant can now think: "Should I search or do I already know this?"

3. **📝 Structured Output**: Using JSON format for consistent action parsing
   - Like having a standard form for the AI to fill out its decisions

4. **🗂️ Context Management**: Handling empty vs. populated context states
   - Knowing when you have enough information vs. when you need more

5. **🏷️ Source Attribution**: Tracking whether answers come from FAQ or general knowledge
   - Always citing your sources - good academic practice! 📚

### 🤖 Understanding Agent Behavior (LLM Zoomcamp Insights)

Your agentic system now exhibits intelligent behavior! 🧠✨

- **📚 For course-specific questions**: Recognizes need to search FAQ database
  - "How do I join the course?" → 🔍 Search FAQ → 📖 Answer from course materials

- **🌍 For general questions**: Uses built-in knowledge without unnecessary searches  
  - "How do I install Python?" → 🧠 Use own knowledge → 💬 Direct answer

- **🎯 Context awareness**: Makes decisions based on available information
  - Knows the difference between "I have info" vs. "I need to find info"

- **💭 Reasoning**: Provides explanations for its chosen actions
  - Not just doing things, but explaining WHY it's doing them

**🎓 LLM Zoomcamp Achievement Unlocked**: You now understand the fundamental difference between basic RAG and agentic RAG! Your assistant doesn't just follow a script - it makes intelligent decisions! 🚀

This foundation prepares you for more sophisticated agentic behaviors in Part 2, where we'll implement iterative search strategies and function calling mechanisms. Get ready to level up! 🚀

---

## ⚡ Part 2: Advanced Implementation - Function Calling and Tool Integration

Welcome to the advanced section of our **LLM Zoomcamp** journey! 🚀 Now we'll build truly sophisticated agentic systems that can think deeply and use multiple tools. Ready to level up? 💪

### 🔄 Agentic Search: Deep Topic Exploration (LLM Zoomcamp Advanced Technique)

Basic agentic RAG makes a single search decision. But what if a student asks "How do I excel in Module 1?" 🤔 A truly intelligent agent should:

1. 🔍 **Perform initial search** with the original question
2. 🧐 **Analyze results** to identify subtopics (Docker, Terraform, assignments)
3. 🎯 **Generate new search queries** for each subtopic  
4. 🔄 **Iterate** until sufficient information is gathered
5. 📝 **Synthesize** a comprehensive answer

This is like doing research for a term paper - you don't just look up one thing, you explore the topic from multiple angles! 📚

#### 🧠 Multi-Iteration Search Strategy (LLM Zoomcamp Deep Dive)

```python
# 🎭 Let's create a super-smart prompt for iterative search
agentic_search_template = """
🎓 You're a course teaching assistant performing DEEP RESEARCH for LLM Zoomcamp students!

QUESTION: {question}

CONTEXT (from previous searches): 
{context}

SEARCH_QUERIES (already performed):
{search_queries}

PREVIOUS_ACTIONS:
{previous_actions}

🎯 Your mission: Provide the most comprehensive answer possible!

You have these superpowers:
1. 🔍 SEARCH - Look up more information in FAQ database
2. 📖 ANSWER_CONTEXT - Answer using all the context you've gathered  
3. 🧠 ANSWER - Answer using your own knowledge

🎯 Guidelines for being an excellent research assistant:
- 🚫 Don't repeat previous search queries (avoid duplicates!)
- 🎪 Generate diverse, specific search terms for deep exploration
- 🔢 Don't exceed {max_iterations} iterations (current: {iteration_number})
- ⏰ If max iterations reached, provide your best possible answer

📋 Output format for SEARCH:
{{
"action": "SEARCH",
"reasoning": "Why you need more information",
"keywords": ["specific_query1", "detailed_query2", "focused_query3"]
}}

📋 Output format for ANSWER_CONTEXT:
{{
"action": "ANSWER_CONTEXT", 
"answer": "Comprehensive answer based on all gathered context",
"source": "CONTEXT"
}}

📋 Output format for ANSWER:
{{
"action": "ANSWER",
"answer": "Answer using your knowledge", 
"source": "OWN_KNOWLEDGE"
}}
""".strip()

print("🧠 Created advanced iterative search prompt!")
print("✨ Now our assistant can do PhD-level research! 🎓")
```

**🎓 LLM Zoomcamp Explanation**: This prompt turns ChatGPT into a research assistant who:
- 🔍 Keeps track of what it has already searched
- 🎯 Generates new, specific search terms based on what it learned
- 🔄 Continues until it has enough information
- 📝 Synthesizes everything into a comprehensive answer

```python
# 🧹 Helper function to remove duplicate search results
def dedup_search_results(results):
    """
    🧹 Remove duplicate search results based on document ID.
    
    Think of this as removing duplicate books from your reading list!
    
    Args:
        results (list): Search results that might have duplicates
        
    Returns:
        list: Clean list with no duplicates
    """
    seen = set()         # Keep track of what we've seen
    deduplicated = []    # Our clean list
    
    for result in results:
        doc_id = result['_id']    # Each document has a unique ID
        if doc_id not in seen:    # If we haven't seen this document before
            seen.add(doc_id)      # Remember that we've seen it
            deduplicated.append(result)  # Add it to our clean list
    
    return deduplicated

print("🧹 Created deduplication function - no more duplicate results!")
```

**🎓 LLM Zoomcamp Explanation**: When we do multiple searches, we might get the same document multiple times. This function is like organizing your bookshelf - removing duplicate books so you don't read the same thing twice! 📚

```python
def agentic_search(question, max_iterations=3):
    """
    🔄 Perform iterative agentic search with multiple query refinements.
    
    This is our PhD-level research assistant that keeps digging deeper!
    
    Args:
        question (str): The student's complex question
        max_iterations (int): Maximum number of search rounds
        
    Returns:
        dict: Comprehensive answer with source attribution
    """
    
    # 📊 Initialize our research tracking variables
    search_queries = []     # All queries we've tried
    search_results = []     # All documents we've found
    previous_actions = []   # History of what we've done
    
    print(f"🔬 Starting deep research on: {question}")
    print(f"🎯 Maximum {max_iterations} research iterations")
    
    # 🔄 Main research loop
    for iteration in range(max_iterations + 1):
        print(f"\n🔍 === RESEARCH ITERATION {iteration} ===")
        
        # 📝 Step 1: Build current context from all our findings
        context = build_context(search_results)
        context_preview = context[:150] + "..." if len(context) > 150 else context
        print(f"📚 Current context: {context_preview}")
        
        # 🎭 Step 2: Create prompt with current research state
        prompt = agentic_search_template.format(
            question=question,
            context=context if context else "EMPTY",
            search_queries="\n".join(search_queries),
            previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
            max_iterations=max_iterations,
            iteration_number=iteration
        )
        
        # 🤖 Step 3: Get ChatGPT's research decision
        print("🤔 Asking research assistant what to do next...")
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        
        print(f"🎯 Decision: {answer['action']}")
        if 'reasoning' in answer:
            print(f"💭 Reasoning: {answer['reasoning']}")
        
        # 📋 Step 4: Record this action in our research log
        previous_actions.append(answer)
        
        # 🔍 Step 5: Handle different research actions
        if answer['action'] == 'SEARCH':
            keywords = answer['keywords']
            print(f"🔍 New search keywords: {keywords}")
            
            # Add new keywords to our search history
            search_queries.extend(keywords)
            
            # Perform searches for each keyword
            print("📚 Searching FAQ database...")
            for keyword in keywords:
                results = search(keyword)
                search_results.extend(results)
                print(f"  📄 '{keyword}': found {len(results)} results")
            
            # Remove duplicate documents
            search_results = dedup_search_results(search_results)
            print(f"📊 Total unique documents: {len(search_results)}")
            
        else:
            # Either ANSWER_CONTEXT or ANSWER - research is complete!
            print("✅ Research complete! Providing final answer...")
            final_answer = answer['answer']
            print(f"📝 Final answer preview: {final_answer[:200]}...")
            return answer
    
    # 🚨 If we've exhausted iterations, force an answer
    print("⏰ Maximum research iterations reached!")
    print("📝 Providing best possible answer with current information...")
    
    final_context = build_context(search_results)
    final_prompt = f"""
    Based on all the research gathered, provide a comprehensive answer to: {question}
    
    Research Context: {final_context}
    
    Please synthesize all the information into a helpful, detailed response.
    """
    
    final_answer = llm(final_prompt)
    
    return {
        "action": "ANSWER_CONTEXT",
        "answer": final_answer,
        "source": "CONTEXT"
    }

# 🧪 Let's test our advanced research assistant!
print("🧪 Testing LLM Zoomcamp Advanced Research Assistant!")
print("=" * 60)

test_question = "What do I need to do to be successful in Module 1 of the data engineering course?"
result = agentic_search(test_question)

print(f"\n🎓 === FINAL RESEARCH RESULTS ===")
print(f"❓ Question: {test_question}")
print(f"🏷️ Source: {result['source']}")
print(f"📝 Answer: {result['answer']}")
```

**🎓 LLM Zoomcamp Explanation**: Our advanced research assistant is like having a PhD student who:

1. 🔍 **Starts with one search** but realizes the topic is complex
2. 🧐 **Analyzes what they found** and identifies knowledge gaps
3. 🎯 **Generates specific new searches** to fill those gaps
4. 🔄 **Repeats this process** until they have comprehensive information
5. 📝 **Synthesizes everything** into one great answer

It's like writing a research paper - you don't just use one source! 📚✨

### 🛠️ OpenAI Function Calling: Structured Tool Integration (LLM Zoomcamp Pro Level)

Great news! 🎉 OpenAI provides a more elegant way to handle tool integration called **Function Calling**. Instead of manually parsing JSON responses, OpenAI's API handles tool descriptions and invocation requests automatically! 

Think of it as upgrading from manual gear shifting to automatic transmission! 🚗⚡

#### 🔧 Defining Tools with Function Calling (LLM Zoomcamp Tools Workshop)

```python
# 🔍 Step 1: Define our search tool specification
# This is like creating a manual that tells ChatGPT how to use our search function
search_tool = {
    "type": "function",                    # This is a function tool
    "name": "search",                      # What to call it
    "description": "Search the LLM Zoomcamp FAQ database for course-related information",
    "parameters": {                        # What inputs it needs
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ (e.g., 'Docker setup', 'course enrollment')"
            }
        },
        "required": ["query"],             # Which inputs are mandatory
        "additionalProperties": False      # Don't accept other inputs
    }
}

print("🔍 Created search tool specification!")
print("📋 ChatGPT now knows exactly how to use our search function!")

# 📝 Step 2: Define an add_entry tool for expanding the FAQ
add_entry_tool = {
    "type": "function", 
    "name": "add_entry",
    "description": "Add a new question-answer entry to the LLM Zoomcamp FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to be added to the FAQ (e.g., 'How do I reset my password?')"
            },
            "answer": {
                "type": "string", 
                "description": "The answer to the question (should be helpful and detailed)"
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }
}

print("📝 Created add_entry tool specification!")
print("✨ Now ChatGPT can help expand our FAQ database!")

# 🏗️ Step 3: Implement the add_entry function
def add_entry(question, answer):
    """
    📝 Add a new entry to the LLM Zoomcamp FAQ database.
    
    This function expands our knowledge base with user contributions!
    
    Args:
        question (str): The question to add
        answer (str): The answer to add
        
    Returns:
        str: Confirmation message
    """
    # 📄 Create a new document
    doc = {
        'question': question,
        'text': answer, 
        'section': 'user_added',                    # Mark as user-contributed
        'course': 'data-engineering-zoomcamp'       # Assign to course
    }
    
    # 📚 Add to our search index
    index.append(doc)
    print(f"✅ Added new FAQ entry: {question}")
    
    return f"Successfully added FAQ entry: {question}"

print("🏗️ Implemented add_entry function!")
print("🎓 Students can now contribute to the LLM Zoomcamp knowledge base!")
```

**🎓 LLM Zoomcamp Explanation**: Tool specifications are like instruction manuals for ChatGPT:

- 📋 **name**: What to call the tool
- 📝 **description**: When and why to use it  
- ⚙️ **parameters**: What information it needs
- ✅ **required**: Which parameters are mandatory

It's like giving someone a recipe - they need to know what ingredients are required vs. optional! 👨‍🍳

#### ⚡ Implementing Function Calling Logic (LLM Zoomcamp Advanced Integration)

Now let's build the smart system that can automatically execute functions when ChatGPT requests them! 🤖

```python
def execute_function_call(function_call):
    """
    ⚡ Execute a function call from OpenAI and return the result.
    
    This is like having a personal assistant who can actually DO things,
    not just talk about them!
    
    Args:
        function_call: OpenAI function call object
        
    Returns:
        dict: Formatted function call result for chat history
    """
    
    # 🏷️ Step 1: Extract function details from OpenAI's request
    function_name = function_call.name                          # Which function to call
    arguments = json.loads(function_call.arguments)            # What parameters to use
    
    print(f"🔧 Executing function: {function_name}")
    print(f"📋 With arguments: {arguments}")
    
    # 🗂️ Step 2: Create our function lookup table
    # This is like a phone book for our functions!
    available_functions = {
        'search': search,           # Maps 'search' name to search function
        'add_entry': add_entry      # Maps 'add_entry' name to add_entry function
    }
    
    # ✅ Step 3: Check if we know this function
    if function_name in available_functions:
        function = available_functions[function_name]   # Get the actual function
        result = function(**arguments)                  # Call it with the arguments
        
        print(f"✅ Function executed successfully!")
        print(f"📤 Result: {str(result)[:100]}...")     # Show preview of result
        
        # 📦 Step 4: Package the result for OpenAI
        return {
            "type": "function_call_output",
            "call_id": function_call.call_id,           # OpenAI needs this ID to track calls
            "output": json.dumps(result, indent=2) if not isinstance(result, str) else result
        }
    else:
        # 🚨 Error handling: Unknown function
        error_msg = f"❌ Unknown function: {function_name}"
        print(error_msg)
        raise ValueError(error_msg)

print("⚡ Function execution system ready!")
print("🤖 ChatGPT can now request actions and we'll execute them!")
```

**🎓 LLM Zoomcamp Explanation**: This function is like having a butler who:

1. 👂 **Listens** to ChatGPT's requests ("Please search for X")
2. 🔍 **Looks up** the right function to call  
3. ⚡ **Executes** the function with the right parameters
4. 📦 **Reports back** the results to ChatGPT

It's the bridge between ChatGPT's requests and actual actions! 🌉

```python
def chat_with_function_calling(question, tools=None, max_iterations=5):
    """
    🎯 Main chat function using OpenAI function calling API.
    
    This is our complete LLM Zoomcamp assistant with function calling superpowers!
    
    Args:
        question (str): The student's question
        tools (list): Available tools (default: search and add_entry)
        max_iterations (int): Maximum function calling rounds
        
    Returns:
        str: Final assistant response
    """
    
    # 🛠️ Step 1: Set up default tools if none provided
    if tools is None:
        tools = [search_tool, add_entry_tool]
        print("🛠️ Using default tools: search and add_entry")
    
    # 🎭 Step 2: Create our LLM Zoomcamp system prompt
    developer_prompt = """
🎓 You're a course teaching assistant for the LLM Zoomcamp!

Your superpowers:
1. 🔍 Search the FAQ database when students have course-specific questions
2. 🧠 Use your own knowledge for general programming/tech questions  
3. 📝 Add new entries to FAQ when students explicitly request it

🎯 Guidelines:
- When searching FAQ, you can make multiple queries to explore topics deeply
- Always provide helpful, detailed, and encouraging responses
- If you search and find relevant info, use it! If not, use your knowledge
- Be a supportive learning companion for LLM Zoomcamp students! 😊
""".strip()
    
    # 💬 Step 3: Initialize the conversation
    chat_messages = [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": question}
    ]
    
    print(f"🎯 Processing question: {question}")
    print(f"🔄 Maximum {max_iterations} function calling iterations")
    
    # 🔄 Step 4: Main function calling loop
    for iteration in range(max_iterations):
        print(f"\n🔄 --- Iteration {iteration + 1} ---")
        
        # 📞 Make API call to OpenAI with our tools
        response = client.responses.create(
            model='gpt-4o-mini',
            input=chat_messages,
            tools=tools
        )
        
        has_function_calls = False
        
        # 📥 Step 5: Process each part of OpenAI's response
        for item in response.output:
            chat_messages.append(item)      # Always add to conversation history
            
            if item.type == 'function_call':
                # 🔧 ChatGPT wants to use a tool!
                args_preview = json.loads(item.arguments)
                print(f"🔧 Function call: {item.name}({args_preview})")
                
                # Execute the function
                result = execute_function_call(item)
                chat_messages.append(result)
                
                print(f"📤 Result preview: {result['output'][:150]}...")
                has_function_calls = True
                
            elif item.type == 'message':
                # 💬 ChatGPT is giving us a final answer!
                response_text = item.content[0].text
                print(f"💬 Assistant response: {response_text[:150]}...")
                
                # If we got a message and no function calls, we're done!
                if not has_function_calls:
                    return response_text
        
        # If no function calls in this iteration, we're done
        if not has_function_calls:
            break
    
    return "⏰ Maximum iterations reached - please try a simpler question!"

# 🧪 Let's test our LLM Zoomcamp function calling system!
print("\n🧪 === Testing LLM Zoomcamp Function Calling System ===")

print("\n📚 Test 1: Course-specific question")
response1 = chat_with_function_calling("How do I prepare for the LLM Zoomcamp course?")
print(f"✅ Final answer: {response1[:200]}...")

print("\n🔄 Test 2: Complex question requiring multiple searches")
response2 = chat_with_function_calling("Tell me everything about succeeding in Module 1")
print(f"✅ Final answer: {response2[:200]}...")
```

**🎓 LLM Zoomcamp Explanation**: Our function calling system is like having a smart research assistant who:

1. 🎯 **Gets a question** from a student
2. 🤔 **Thinks** about whether they need to search or can answer directly
3. 🔧 **Uses tools** (search, add_entry) when needed
4. 🔄 **Continues** using tools until they have enough information
5. 💬 **Provides** a comprehensive final answer

It's like upgrading from a simple Q&A bot to a intelligent tutor! 🎓✨

print("\n=== Testing with Multiple Calls ===") 
response2 = chat_with_function_calling("Tell me everything about module 1 success strategies")

print("\n=== Testing Add Entry ===")
response3 = chat_with_function_calling("Add this to the FAQ: Question: What IDE should I use? Answer: VS Code is recommended for this course.")
```

### 💬 Building a Conversational Agent (LLM Zoomcamp Interactive Experience)

Now let's create a full conversational experience! This will be like having a persistent study buddy who remembers your conversation and can help with multiple questions! 🎓💬

#### 🔄 Two-Loop Architecture (LLM Zoomcamp Advanced Pattern)

```python
def run_conversational_agent():
    """
    🎭 Run our LLM Zoomcamp conversational agent!
    
    This creates a persistent chat experience with two smart loops:
    - 🔄 Outer loop: Handle multiple user questions (conversation continues)
    - ⚡ Inner loop: Process function calls until getting a final response
    
    It's like having office hours with a TA who never gets tired! 🎓
    """
    
    # 🛠️ Set up our tools
    tools = [search_tool, add_entry_tool]
    
    # 🎭 Create an engaging system prompt for ongoing conversation
    developer_prompt = """
🎓 You're a friendly LLM Zoomcamp teaching assistant having an ongoing conversation with a student!

Your conversation superpowers:
- 🔍 Search the FAQ database for course-specific questions
- 🧠 Use your knowledge for general programming/tech questions
- 💬 Keep the conversation engaging and educational
- 🤔 Ask thoughtful follow-up questions to help students learn deeper
- 📝 Add FAQ entries when students explicitly request it

🎯 Conversation style:
- Be encouraging and supportive (learning can be challenging!)
- Use emojis to make responses friendly and engaging
- At the end of each response, ask a relevant follow-up question
- Remember the conversation context for better continuity
""".strip()
    
    # 💬 Initialize conversation with system prompt
    chat_messages = [
        {"role": "developer", "content": developer_prompt}
    ]
    
    # 🎉 Welcome message
    print("🎓✨ LLM Zoomcamp Teaching Assistant ✨🎓")
    print("Hello! I'm your friendly LLM Zoomcamp TA! 😊")
    print("Ask me anything about the course, and I'll help you succeed! 🚀")
    print("(Type 'stop' when you're ready to end our conversation)")
    print("=" * 50)
    
    # 🔄 Main conversation loop
    while True:  # Outer Q&A loop - keeps conversation going
        # 💭 Get user input
        user_input = input("\n🎓 You: ").strip()
        
        # 🛑 Check if user wants to end conversation
        if user_input.lower() in ['stop', 'quit', 'exit', 'bye']:
            print("\n🎓 LLM Zoomcamp TA: Thanks for studying with me! 📚")
            print("Keep up the great work in your LLM journey! 🚀✨")
            print("See you next time! 👋")
            break
        
        # 🔍 Skip empty inputs
        if not user_input:
            print("💭 (Please ask me a question, or type 'stop' to end)")
            continue
        
        # 💬 Add user message to conversation history
        chat_messages.append({"role": "user", "content": user_input})
        
        # ⚡ Inner loop: Process function calls until final response
        iteration_count = 0
        max_inner_iterations = 8  # Prevent infinite loops
        
        while iteration_count < max_inner_iterations:
            # 📞 Make API call with conversation history and tools
            response = client.responses.create(
                model='gpt-4o-mini',
                input=chat_messages,
                tools=tools
            )
            
            has_function_calls = False
            
            # 📥 Process each part of the response
            for item in response.output:
                chat_messages.append(item)  # Always add to conversation history
                
                if item.type == 'function_call':
                    # 🔧 Assistant is using a tool - execute it quietly
                    try:
                        result = execute_function_call(item)
                        chat_messages.append(result)
                        has_function_calls = True
                        
                        # Optional: Show what tool was used (for transparency)
                        args = json.loads(item.arguments)
                        if item.name == 'search':
                            print(f"🔍 (Searching FAQ for: {args.get('query', 'N/A')})")
                        elif item.name == 'add_entry':
                            print(f"📝 (Adding new FAQ entry)")
                        
                    except Exception as e:
                        print(f"🚨 Tool error: {str(e)}")
                        # Add error to conversation so assistant knows what happened
                        error_result = {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": f"Error: {str(e)}"
                        }
                        chat_messages.append(error_result)
                        has_function_calls = True
                    
                elif item.type == 'message':
                    # 💬 Assistant is giving us a response!
                    response_text = item.content[0].text
                    print(f"\n🤖 LLM Zoomcamp TA: {response_text}")
            
            # 🏁 Exit inner loop if no function calls (conversation turn complete)
            if not has_function_calls:
                break
            
            iteration_count += 1
        
        # 🚨 Safety check for infinite function calling
        if iteration_count >= max_inner_iterations:
            print("\n🤖 LLM Zoomcamp TA: I got a bit carried away with research! 😅")
            print("Could you please rephrase your question or ask something else?")

# 🎮 Ready to start the conversation!
print("🎮 LLM Zoomcamp Conversational Agent Ready!")
print("💡 Uncomment the next line to start chatting:")
print("# run_conversational_agent()")
```

**🎓 LLM Zoomcamp Explanation**: Our conversational agent works like this:

- 🔄 **Outer Loop**: Keeps asking "What's your next question?" (like office hours)
- ⚡ **Inner Loop**: Handles the assistant's thinking process (search, research, respond)
- 💭 **Memory**: Remembers the whole conversation for context
- 🛠️ **Tools**: Can search FAQ and add entries as needed
- 💬 **Persistent**: Continues until you say "stop"

It's like having a study session that can last as long as you need! 📚✨

### 🎨 Enhanced Display and Debugging (LLM Zoomcamp Visual Experience)

For the best user experience in Jupyter notebooks, let's add beautiful, interactive formatting! This makes debugging fun and results easy to read! 🎨✨

```python
from IPython.display import display, HTML
import markdown

def display_function_call(function_call, result):
    """
    🎨 Display function calls with beautiful, collapsible formatting.
    
    This makes debugging feel like exploring a well-organized filing cabinet! 🗂️
    
    Args:
        function_call: OpenAI function call object
        result: Function execution result
    """
    
    # 🎭 Create beautiful, collapsible HTML
    args_str = json.dumps(json.loads(function_call.arguments), indent=2)
    output_str = result['output']
    
    # ✂️ Truncate long outputs for better display
    if len(output_str) > 500:
        output_str = output_str[:500] + "\n... (truncated for display)"
    
    html_content = f"""
    <details style="margin: 15px 0; border: 2px solid #e1e5e9; border-radius: 8px; 
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
        <summary style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                        color: white; padding: 12px 15px; cursor: pointer; border-radius: 6px;
                        font-weight: bold; font-size: 14px;">
            🔧 Function Call: <code style="color: #ffd700;">{function_call.name}</code>
            <span style="float: right; font-size: 12px;">Click to expand 📋</span>
        </summary>
        <div style="padding: 20px; background: white; border-radius: 0 0 6px 6px;">
            <div style="margin-bottom: 15px;">
                <h4 style="color: #495057; margin: 0 0 10px 0; font-size: 14px;">
                    📥 Arguments:
                </h4>
                <pre style="background: #f8f9fa; padding: 12px; border-radius: 4px; 
                           border-left: 4px solid #007bff; overflow-x: auto; font-size: 12px;">{args_str}</pre>
            </div>
            <div>
                <h4 style="color: #495057; margin: 0 0 10px 0; font-size: 14px;">
                    📤 Result:
                </h4>
                <pre style="background: #e8f5e8; padding: 12px; border-radius: 4px; 
                           border-left: 4px solid #28a745; overflow-x: auto; font-size: 12px;">{output_str}</pre>
            </div>
        </div>
    </details>
    """
    
    display(HTML(html_content))

def display_assistant_response(content):
    """
    💬 Display assistant responses with beautiful markdown formatting.
    
    This makes the assistant's responses look professional and easy to read! ✨
    
    Args:
        content (str): Response text from the assistant
    """
    
    # 📝 Convert markdown to HTML for rich formatting
    html_content = markdown.markdown(content)
    
    # 🎨 Wrap in beautiful styling
    formatted_html = f"""
    <div style="background: linear-gradient(135deg, #f0f7ff 0%, #e6f3ff 100%); 
                border-left: 5px solid #007bff; padding: 20px; margin: 15px 0; 
                border-radius: 10px; box-shadow: 0 2px 10px rgba(0,123,255,0.1);">
        <div style="color: #007bff; font-weight: bold; margin-bottom: 15px; 
                    display: flex; align-items: center; font-size: 16px;">
            🤖 LLM Zoomcamp Assistant
            <span style="margin-left: auto; font-size: 12px; color: #6c757d;">
                ✨ Powered by OpenAI
            </span>
        </div>
        <div style="color: #495057; line-height: 1.6;">
            {html_content}
        </div>
    </div>
    """
    
    display(HTML(formatted_html))

def display_thinking(message):
    """
    🤔 Display thinking/processing messages with style.
    
    Args:
        message (str): Thinking message to display
    """
    
    html = f"""
    <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
                border: 1px solid #ffc107; padding: 10px 15px; margin: 10px 0; 
                border-radius: 6px; color: #856404; font-style: italic;">
        🤔 <strong>Assistant is thinking:</strong> {message}
    </div>
    """
    display(HTML(html))

def shorten_text(text, max_length=150):
    """
    ✂️ Shorten long text for display purposes.
    
    Prevents overwhelming displays while keeping important info visible! 
    
    Args:
        text (str): Text to potentially shorten
        max_length (int): Maximum length before truncation
        
    Returns:
        str: Original or shortened text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "... 📝"

def display_search_results(results, query):
    """
    🔍 Display search results in a beautiful, organized format.
    
    Args:
        results (list): Search results from our FAQ
        query (str): The original search query
    """
    
    html = f"""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; 
                padding: 15px; margin: 10px 0;">
        <h4 style="color: #495057; margin: 0 0 15px 0;">
            🔍 Search Results for: "{query}"
        </h4>
        <div style="color: #6c757d; font-size: 14px;">
            Found {len(results)} relevant FAQ entries:
        </div>
    """
    
    for i, result in enumerate(results[:3]):  # Show top 3 results
        html += f"""
        <div style="background: white; border-left: 3px solid #28a745; 
                    padding: 10px; margin: 10px 0; border-radius: 4px;">
            <strong style="color: #495057;">#{i+1}: {result.get('question', 'No question')}</strong><br>
            <span style="color: #6c757d; font-size: 12px;">
                Section: {result.get('section', 'N/A')} | 
                Score: {result.get('score', 'N/A')}
            </span>
        </div>
        """
    
    html += "</div>"
    display(HTML(html))

print("🎨 Beautiful display functions ready!")
print("✨ Your LLM Zoomcamp experience will now look amazing! 🎓")
```

**🎓 LLM Zoomcamp Explanation**: These display functions are like having a professional web designer for your notebook:

- 🎨 **Collapsible Function Calls**: Click to expand and see details
- 💬 **Styled Responses**: Assistant messages look professional  
- 🤔 **Thinking Indicators**: Shows when the assistant is processing
- ✂️ **Smart Truncation**: Prevents overwhelming long outputs
- 🔍 **Search Result Cards**: Organized, easy-to-scan search results

It transforms a plain text experience into something beautiful and engaging! ✨

### 🎓 Key Advances in Part 2 (LLM Zoomcamp Level Up!)

Congratulations! You've just built a sophisticated agentic system! 🎉 Let's celebrate what you've accomplished:

1. **🔄 Iterative Search**: Multiple search rounds with query refinement
   - Your assistant can now research topics like a PhD student! 🎓
   - It doesn't stop at the first search - it keeps digging deeper

2. **⚡ Function Calling API**: Structured tool integration with OpenAI
   - Upgraded from manual JSON parsing to automatic tool handling
   - ChatGPT now knows exactly how to use your functions

3. **🛠️ Tool Composition**: Multiple tools working together (search + add_entry)
   - Your assistant is a multi-tool Swiss Army knife! 🔧
   - Can search existing knowledge AND create new knowledge

4. **💬 Conversational Flow**: Persistent multi-turn conversations
   - Remembers the whole conversation like a real teaching assistant
   - Context carries forward for natural, flowing discussions

5. **🎨 Rich Display**: Enhanced formatting for better user experience
   - Beautiful, interactive displays that make debugging fun
   - Professional-looking responses that are easy to read

6. **⚡ Dynamic Function Execution**: Runtime function lookup and invocation
   - Your system can call functions by name dynamically
   - Easy to add new tools without changing core logic

### 🚀 What Your LLM Zoomcamp Assistant Can Now Do

Your system now exhibits truly sophisticated agentic behavior! 🧠✨

- **🔬 Deep Topic Exploration**: Researches topics from multiple angles
  - "How do I succeed in Module 1?" → Searches for assignments, Docker, tips, troubleshooting

- **🎯 Smart Tool Selection**: Chooses the right tool for each task
  - Course questions → Search FAQ
  - General questions → Use knowledge
  - "Add this to FAQ" → Use add_entry tool

- **💭 Context Preservation**: Remembers conversation history
  - "What about Module 2?" (remembers you were discussing modules)
  - "Can you search for more details?" (knows what to search for)

- **🔄 Autonomous Decision-Making**: Plans and executes multi-step actions
  - Decides when to search, how many times, and when it has enough info
  - Can course-correct based on search results

**🎓 LLM Zoomcamp Achievement Unlocked**: You've built a production-quality agentic assistant! Your system doesn't just follow instructions - it makes intelligent decisions, uses tools effectively, and provides excellent user experience! 🏆

Ready for Part 3? We'll organize this code into professional, production-ready classes and explore advanced frameworks! 🚀

---

## 🏭 Part 3: Production Ready - Object-Oriented Design and Libraries

Welcome to the final phase of your **LLM Zoomcamp** agentic journey! 🎓 Now we'll transform our functional code into professional, production-ready classes that real companies use in their AI systems. Ready to become an enterprise AI developer? 💼✨
### 🏗️ Modularizing with Object-Oriented Design (LLM Zoomcamp Enterprise Level)

As our agentic system grows, we need to organize our code like a professional software company! Let's transform our functions into beautiful, reusable classes. 💼

#### 🛠️ The Tools Management System (LLM Zoomcamp Professional Pattern)

Think of this as creating a "toolbox manager" for our AI assistant:

```python
import inspect
from typing import Callable, Dict, Any, List

class LLMZoomcampTools:
    """
    🛠️ Professional tool manager for LLM Zoomcamp assistants!
    
    This class is like a smart toolbox that:
    - 📋 Keeps track of all available tools
    - 🔍 Automatically generates tool descriptions
    - ⚡ Executes tools when the AI requests them
    - 🧹 Handles errors gracefully
    
    Perfect for building enterprise AI systems! 💼
    """
    
    def __init__(self):
        """🎬 Initialize our professional toolbox!"""
        self.tools = {}                 # function_name -> actual_function
        self.tool_descriptions = []     # OpenAI tool schemas
        print("🛠️ LLM Zoomcamp Tools Manager initialized!")
    
    def add_tool(self, name: str, function: Callable, description: Dict[str, Any]):
        """
        📝 Register a new tool with its description.
        
        Args:
            name: Tool name (like 'search' or 'add_entry')
            function: The actual Python function
            description: OpenAI-compatible tool schema
        """
        self.tools[name] = function
        self.tool_descriptions.append(description)
        print(f"✅ Added tool: {name}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """📋 Get all tool descriptions for OpenAI API."""
        return self.tool_descriptions
    
    def execute_function_call(self, function_call) -> Dict[str, Any]:
        """
        ⚡ Execute a function call professionally with error handling.
        
        This is production-ready code that handles errors gracefully!
        """
        function_name = function_call.name
        
        try:
            arguments = json.loads(function_call.arguments)
            
            if function_name not in self.tools:
                raise ValueError(f"Unknown tool: {function_name}")
            
            # Execute the function
            function = self.tools[function_name]
            result = function(**arguments)
            
            return {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": json.dumps(result, indent=2) if not isinstance(result, str) else result
            }
            
        except Exception as e:
            # Professional error handling
            return {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": f"Error executing {function_name}: {str(e)}"
            }

print("🏗️ Professional Tools Management System ready!")
```

#### 🎓 LLM Zoomcamp Professional Assistant Class

```python
class LLMZoomcampAssistant:
    """
    🎓 Professional LLM Zoomcamp Teaching Assistant!
    
    This is enterprise-level code that real companies use for AI assistants.
    Features:
    - 🛠️ Tool management
    - 💬 Conversation handling  
    - 🚨 Error handling
    - 📊 Logging and monitoring
    - 🎨 Beautiful user interface
    """
    
    def __init__(self, tools: LLMZoomcampTools, client):
        """🎬 Initialize our professional assistant."""
        self.tools = tools
        self.client = client
        self.conversation_history = []
        
        # 🎭 Professional system prompt
        self.system_prompt = """
🎓 You're a professional LLM Zoomcamp Teaching Assistant!

Your mission: Help students succeed in their learning journey with:
- 🔍 Intelligent FAQ searching
- 🧠 Expert knowledge sharing
- 📝 Knowledge base expansion
- 💬 Engaging, supportive conversations

Always be encouraging, thorough, and educational! 🌟
""".strip()
        
        print("🎓 LLM Zoomcamp Professional Assistant ready!")
    
    def chat(self, user_message: str) -> str:
        """
        💬 Handle a single chat interaction professionally.
        
        This method handles the complete conversation flow with
        proper error handling and logging.
        """
        try:
            # Add user message to history
            self.conversation_history.extend([
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            # Process with function calling
            max_iterations = 8
            for iteration in range(max_iterations):
                response = self.client.responses.create(
                    model='gpt-4o-mini',
                    input=self.conversation_history,
                    tools=self.tools.get_tools()
                )
                
                has_function_calls = False
                
                for item in response.output:
                    self.conversation_history.append(item)
                    
                    if item.type == 'function_call':
                        result = self.tools.execute_function_call(item)
                        self.conversation_history.append(result)
                        has_function_calls = True
                        
                    elif item.type == 'message':
                        if not has_function_calls:
                            return item.content[0].text
                
                if not has_function_calls:
                    break
            
            return "I apologize, but I couldn't complete your request. Please try again!"
            
        except Exception as e:
            return f"🚨 Sorry, I encountered an error: {str(e)}"

# 🎉 Ready to create your professional assistant!
print("🎉 LLM Zoomcamp Professional Classes ready!")
```

### 🚀 Quick Start: Building Your Professional LLM Zoomcamp Assistant

```python
def create_professional_llm_zoomcamp_assistant():
    """
    🚀 Factory function to create a complete, professional assistant!
    
    This is how real companies structure their AI systems!
    """
    
    # 🛠️ Step 1: Create tools manager
    tools = LLMZoomcampTools()
    
    # 📝 Step 2: Add our search and add_entry tools
    tools.add_tool("search", search, search_tool)
    tools.add_tool("add_entry", add_entry, add_entry_tool)
    
    # 🎓 Step 3: Create the assistant
    assistant = LLMZoomcampAssistant(tools, client)
    
    print("✅ Professional LLM Zoomcamp Assistant created!")
    return assistant

# 🧪 Test your professional assistant!
def test_professional_assistant():
    """🧪 Test our professional-grade system!"""
    assistant = create_professional_llm_zoomcamp_assistant()
    
    # Test questions
    questions = [
        "How do I succeed in Module 1?",
        "What's the best way to learn machine learning?",
        "Can you add this to the FAQ: Question: How to debug Python? Answer: Use print statements and debugger."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🧪 Test {i}: {question}")
        response = assistant.chat(question)
        print(f"✅ Response: {response[:200]}...")

print("🚀 Professional LLM Zoomcamp system ready!")
print("💡 Run test_professional_assistant() to see it in action!")
```

### 🎓 Beyond LLM Zoomcamp: Production Frameworks

Your **LLM Zoomcamp** journey has prepared you for real-world frameworks! 🌟

- **🔥 PydanticAI**: Type-safe agents with validation
- **🦜 LangChain**: Complex workflow orchestration
- **🤖 OpenAI Assistant API**: Hosted agent infrastructure
- **🔧 Custom Frameworks**: Build your own specialized systems

**🎓 LLM Zoomcamp Achievement**: You now understand the fundamentals that power all these frameworks! 🏆

---

## 🎉 Conclusion: Your LLM Zoomcamp Agentic Journey

Congratulations, **LLM Zoomcamp** graduate! 🎓 You've completed an incredible journey from basic RAG to sophisticated agentic systems! 

### 🏆 What You've Accomplished

1. **🏗️ Built Basic RAG**: Search + Context + LLM Pipeline
2. **🧠 Made It Agentic**: Added decision-making capabilities  
3. **🔄 Implemented Iterative Search**: Deep topic exploration
4. **⚡ Mastered Function Calling**: Professional tool integration
5. **💬 Created Conversational Agents**: Multi-turn interactions
6. **🎨 Added Beautiful UI**: Production-quality displays
7. **🏭 Designed Enterprise Classes**: Professional code organization

### 🚀 Your Next Steps

- **Experiment** with different prompting strategies 🎭
- **Build** domain-specific agents for your projects 🔨  
- **Explore** frameworks like PydanticAI and LangChain 🦜
- **Contribute** to open-source agent projects 🤝
- **Share** your LLM Zoomcamp knowledge with others! 📚

### 🌟 Final Words

The future of AI is agentic, and you're now equipped to be part of building it! Your **LLM Zoomcamp** journey has given you the foundation to create truly intelligent AI assistants that can think, decide, and act autonomously.

Keep experimenting, keep learning, and keep building amazing things! 🚀✨

---

## 📚 Resources for Continued Learning

- **🎓 LLM Zoomcamp**: [Continue your learning journey](https://github.com/DataTalksClub/llm-zoomcamp)
- **Workshop Repository**: [rag-agents-workshop](https://github.com/alexeygrigorev/rag-agents-workshop)
- **🧸 toy_AI_kit Library**: [Educational agent framework](https://github.com/alexeygrigorev/toyaikit)
- **🤖 OpenAI Function Calling**: [Official documentation](https://platform.openai.com/docs/guides/function-calling)
- **🔥 PydanticAI**: [Type-safe agent framework](https://ai.pydantic.dev/)

*Happy building, LLM Zoomcamp graduate! 🎓🚀*
    """
    
    def __init__(self):
        self.tools = {}  # function_name -> function_object
        self.tool_descriptions = []  # OpenAI tool schemas
    
    def add_tool(self, name: str, function: Callable, description: Dict[str, Any]):
        """
        Register a tool with its function and description.
        
        Args:
            name: Tool name (must match function name in description)
            function: The actual Python function to execute
            description: OpenAI tool schema describing the function
        """
        self.tools[name] = function
        self.tool_descriptions.append(description)
    
    def auto_add_tool(self, function: Callable):
        """
        Automatically generate tool description from function docstring and type hints.
        
        Args:
            function: Function with proper docstring and type annotations
        """
        name = function.__name__
        signature = inspect.signature(function)
        docstring = function.__doc__ or ""
        
        # Extract description from docstring
        lines = docstring.strip().split('\n')
        description = lines[0] if lines else f"Execute {name} function"
        
        # Build parameters schema from function signature
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = "string"  # Default type
            param_desc = f"Parameter {param_name}"
            
            # Infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
            
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
            }
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        tool_schema = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", 
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
        }
        
        self.add_tool(name, function, tool_schema)
    
    def add_tools_from_instance(self, instance):
        """
        Automatically add all callable methods from an instance as tools.
        Useful for grouping related tools in a class.
        
        Args:
            instance: Object instance containing tool methods
        """
        for attr_name in dir(instance):
            attr = getattr(instance, attr_name)
            
            # Skip private methods and non-callable attributes
            if attr_name.startswith('_') or not callable(attr):
                continue
            
            # Auto-generate tool description
            self.auto_add_tool(attr)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tool descriptions for OpenAI API."""
        return self.tool_descriptions
    
    def execute_function_call(self, function_call) -> Dict[str, Any]:
        """
        Execute a function call from OpenAI and return formatted result.
        
        Args:
            function_call: OpenAI function call object
            
        Returns:
            Formatted function call result for chat history
        """
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)
        
        if function_name not in self.tools:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Execute the function
        function = self.tools[function_name]
        result = function(**arguments)
        
        return {
            "type": "function_call_output",
            "call_id": function_call.call_id,
            "output": json.dumps(result, indent=2) if not isinstance(result, str) else result
        }
```

#### Domain-Specific Tool Collections

```python
class CourseFAQTools:
    """
    Collection of tools specific to course FAQ management.
    Groups related functionality and manages shared dependencies.
    """
    
    def __init__(self, index):
        """
        Initialize with FAQ index dependency.
        
        Args:
            index: Search index containing FAQ documents
        """
        self.index = index
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the FAQ database for relevant entries matching the query.
        
        Args:
            query: The search query string provided by the user
            
        Returns:
            List of search results with relevance information
        """
        boost = {'question': 3.0, 'section': 0.5}
        
        results = self.index.search(
            query=query,
            filter_dict={'course': 'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5,
            output_ids=True
        )
        
        return results
    
    def add_entry(self, question: str, answer: str) -> str:
        """
        Add a new question-answer entry to the FAQ database.
        
        Args:
            question: The question text to be added to the index
            answer: The answer or explanation corresponding to the question
            
        Returns:
            Confirmation message
        """
        doc = {
            'question': question,
            'text': answer,
            'section': 'user_added',
            'course': 'data-engineering-zoomcamp'
        }
        
        self.index.append(doc)
        return f"Successfully added FAQ entry: {question}"
    
    def search_multiple(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Perform multiple searches and combine results.
        
        Args:
            queries: List of search query strings
            
        Returns:
            Combined and deduplicated search results
        """
        all_results = []
        
        for query in queries:
            results = self.search(query)
            all_results.extend(results)
        
        # Deduplicate by document ID
        seen_ids = set()
        deduplicated = []
        
        for result in all_results:
            doc_id = result.get('_id')
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduplicated.append(result)
        
        return deduplicated
```

#### User Interface Abstraction

```python
class IPythonChatInterface:
    """
    Handles user interaction and output display in Jupyter notebooks.
    Provides rich formatting and collapsible function call details.
    """
    
    def get_input(self, prompt: str = "You: ") -> str:
        """Get user input with custom prompt."""
        return input(prompt).strip()
    
    def display_chat_ended(self):
        """Display chat end message."""
        print("\n✨ Chat ended. Thank you for using the course assistant!")
    
    def display_function_call(self, function_call, result: Dict[str, Any]):
        """
        Display function call details in a collapsible format.
        
        Args:
            function_call: OpenAI function call object
            result: Function execution result
        """
        args_str = json.dumps(json.loads(function_call.arguments), indent=2)
        output_str = result['output']
        
        # Truncate long outputs for display
        if len(output_str) > 300:
            output_str = output_str[:300] + "\n... (truncated)"
        
        html_content = f"""
        <details style="margin: 10px 0; border: 1px solid #ddd; border-radius: 5px;">
        <summary style="background-color: #f8f9fa; padding: 10px; cursor: pointer;">
            <strong>🔧 Function Call:</strong> <code>{function_call.name}</code>
        </summary>
        <div style="padding: 15px;">
            <div style="margin-bottom: 10px;">
                <strong>Arguments:</strong>
                <pre style="background-color: #f1f1f1; padding: 10px; border-radius: 3px; overflow-x: auto;">{args_str}</pre>
            </div>
            <div>
                <strong>Result:</strong>
                <pre style="background-color: #e8f5e8; padding: 10px; border-radius: 3px; overflow-x: auto;">{output_str}</pre>
            </div>
        </div>
        </details>
        """
        
        display(HTML(html_content))
    
    def display_response(self, content: str):
        """
        Display assistant response with markdown formatting.
        
        Args:
            content: Response text from the assistant
        """
        # Convert markdown to HTML
        html_content = markdown.markdown(content)
        
        formatted_html = f"""
        <div style="background-color: #f0f7ff; border-left: 4px solid #0066cc; 
                    padding: 15px; margin: 10px 0; border-radius: 5px;">
            <div style="color: #0066cc; font-weight: bold; margin-bottom: 10px;">
                🤖 Course Assistant
            </div>
            {html_content}
        </div>
        """
        
        display(HTML(formatted_html))
    
    def display_thinking(self, message: str):
        """Display thinking/processing message."""
        print(f"🤔 {message}")
```

#### Main Chat Assistant Class

```python
class ChatAssistant:
    """
    Main orchestrator for agentic chat conversations.
    Manages the conversation flow, tool execution, and response generation.
    """
    
    def __init__(self, tools: Tools, developer_prompt: str, 
                 chat_interface: IPythonChatInterface, client):
        """
        Initialize the chat assistant.
        
        Args:
            tools: Tools instance managing available functions
            developer_prompt: System prompt defining assistant behavior
            chat_interface: Interface for user interaction and display
            client: OpenAI client for API calls
        """
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.interface = chat_interface
        self.client = client
        self.chat_messages = [
            {"role": "developer", "content": developer_prompt}
        ]
    
    def gpt(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Make OpenAI API call with current tools.
        
        Args:
            messages: Chat message history
            
        Returns:
            OpenAI response object
        """
        return self.client.responses.create(
            model='gpt-4o-mini',
            input=messages,
            tools=self.tools.get_tools()
        )
    
    def run(self):
        """
        Run the main chat loop with two nested loops:
        - Outer loop: Handle user questions
        - Inner loop: Process function calls until final response
        """
        self.interface.display_response(
            "Hello! I'm your course assistant. Ask me anything about the course, "
            "or type 'stop' to end our conversation."
        )
        
        while True:  # Outer Q&A loop
            try:
                user_input = self.interface.get_input()
                
                if user_input.lower() in ['stop', 'quit', 'exit']:
                    self.interface.display_chat_ended()
                    break
                
                if not user_input:
                    continue
                
                # Add user message to conversation history
                user_message = {"role": "user", "content": user_input}
                self.chat_messages.append(user_message)
                
                # Inner loop: Process until we get a final response
                iteration_count = 0
                max_iterations = 10
                
                while iteration_count < max_iterations:
                    response = self.gpt(self.chat_messages)
                    has_function_calls = False
                    has_messages = False
                    
                    # Process each item in the response
                    for item in response.output:
                        self.chat_messages.append(item)
                        
                        if item.type == 'function_call':
                            # Execute function and add result to conversation
                            try:
                                result = self.tools.execute_function_call(item)
                                self.chat_messages.append(result)
                                self.interface.display_function_call(item, result)
                                has_function_calls = True
                            except Exception as e:
                                error_result = {
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": f"Error: {str(e)}"
                                }
                                self.chat_messages.append(error_result)
                        
                        elif item.type == 'message':
                            # Display the assistant's response
                            content = item.content[0].text
                            self.interface.display_response(content)
                            has_messages = True
                    
                    # Exit inner loop if we got a message and no function calls
                    if has_messages and not has_function_calls:
                        break
                    
                    iteration_count += 1
                
                if iteration_count >= max_iterations:
                    self.interface.display_response(
                        "I apologize, but I've reached the maximum number of processing steps. "
                        "Please try rephrasing your question."
                    )
            
            except KeyboardInterrupt:
                self.interface.display_chat_ended()
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Please try again.")
```

#### Putting It All Together

```python
def create_course_assistant():
    """
    Factory function to create a fully configured course assistant.
    
    Returns:
        Configured ChatAssistant instance ready to run
    """
    # Initialize components
    tools = Tools()
    faq_tools = CourseFAQTools(index)  # Assuming index is available globally
    
    # Auto-register all FAQ tools
    tools.add_tools_from_instance(faq_tools)
    
    # Define system behavior
    developer_prompt = """
You're an intelligent course teaching assistant for an online data engineering course.

Your capabilities:
- Search the course FAQ database to answer specific course questions
- Add new entries to the FAQ when explicitly requested
- Use your general knowledge for questions outside the course scope
- Provide comprehensive, helpful responses

Guidelines:
- When students ask about course-specific topics (enrollment, assignments, deadlines, technical setup), search the FAQ first
- For complex topics, perform multiple searches with different query terms to gather comprehensive information
- Always provide detailed, actionable answers
- When appropriate, ask follow-up questions to better assist the student
- If asked to add something to the FAQ, extract the question and answer clearly

Remember: You're here to help students succeed in their learning journey!
""".strip()
    
    # Create interface and assistant
    chat_interface = IPythonChatInterface()
    
    assistant = ChatAssistant(
        tools=tools,
        developer_prompt=developer_prompt,
        chat_interface=chat_interface,
        client=client
    )
    
    return assistant

# Usage example
def run_course_assistant():
    """Run the course assistant application."""
    assistant = create_course_assistant()
    assistant.run()

# To start the assistant:
# run_course_assistant()
```

### Introduction to toy_AI_kit Library

The modular architecture we've built forms the foundation of the **toy_AI_kit** library - an educational framework for understanding agentic systems.

#### Library Structure

```python
# Example of how toy_AI_kit is organized

from toy_ai_kit.chat import ChatAssistant
from toy_ai_kit.tools import Tools
from toy_ai_kit.ipython import IPythonChatInterface
from toy_ai_kit.lm import OpenAIProvider  # Provider-agnostic LLM interface

# Simple usage with the library
def quick_start_with_library():
    """Demonstrate toy_AI_kit usage."""
    
    # Initialize provider-agnostic LLM interface
    llm_provider = OpenAIProvider(model='gpt-4o-mini')
    
    # Create tools manager
    tools = Tools()
    
    # Add tools with automatic schema generation
    @tools.tool
    def search_docs(query: str) -> str:
        """Search documentation for relevant information."""
        return search(query)
    
    @tools.tool  
    def add_knowledge(question: str, answer: str) -> str:
        """Add new knowledge to the database."""
        return add_entry(question, answer)
    
    # Create assistant
    assistant = ChatAssistant(
        llm=llm_provider,
        tools=tools,
        system_prompt="You're a helpful assistant.",
        interface=IPythonChatInterface()
    )
    
    # Run
    assistant.run()
```

#### Advanced Features and Patterns

```python
class AdvancedFAQTools(CourseFAQTools):
    """
    Extended FAQ tools with advanced capabilities.
    """
    
    def semantic_search(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform semantic search with similarity threshold.
        
        Args:
            query: Search query
            threshold: Minimum similarity score
            
        Returns:
            Filtered results above threshold
        """
        results = self.search(query)
        
        # In a real implementation, you would filter by semantic similarity
        # This is a simplified example
        return [r for r in results if r.get('score', 0) > threshold]
    
    def get_topic_overview(self, topic: str) -> str:
        """
        Get comprehensive overview of a topic by searching multiple angles.
        
        Args:
            topic: Topic to research
            
        Returns:
            Comprehensive topic summary
        """
        # Generate multiple search queries for the topic
        search_queries = [
            f"{topic} overview",
            f"{topic} getting started", 
            f"{topic} best practices",
            f"{topic} troubleshooting",
            f"how to {topic}"
        ]
        
        all_results = self.search_multiple(search_queries)
        
        # Combine results into a comprehensive overview
        if not all_results:
            return f"No information found for topic: {topic}"
        
        overview = f"# {topic.title()} Overview\n\n"
        
        for i, result in enumerate(all_results[:5]):  # Limit to top 5 results
            overview += f"## Point {i+1}: {result.get('question', 'Information')}\n"
            overview += f"{result.get('text', 'No details available')}\n\n"
        
        return overview
    
    def validate_entry(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Validate a new FAQ entry before adding it.
        
        Args:
            question: Proposed question
            answer: Proposed answer
            
        Returns:
            Validation result with suggestions
        """
        # Check for duplicate questions
        existing_results = self.search(question)
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "suggestions": []
        }
        
        # Check for very similar existing questions
        for result in existing_results[:3]:
            similarity_score = self._calculate_similarity(question, result.get('question', ''))
            if similarity_score > 0.8:  # High similarity threshold
                validation["warnings"].append(
                    f"Similar question exists: {result.get('question')}"
                )
        
        # Validate answer length
        if len(answer.split()) < 10:
            validation["suggestions"].append(
                "Consider providing a more detailed answer (current: {len(answer.split())} words)"
            )
        
        return validation
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity (simplified implementation).
        In production, use proper NLP similarity measures.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split()) 
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
```

### Production Considerations

#### Error Handling and Resilience

```python
class RobustChatAssistant(ChatAssistant):
    """
    Enhanced ChatAssistant with production-ready error handling.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def gpt_with_retry(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Make OpenAI API call with automatic retry logic.
        """
        import time
        
        for attempt in range(self.max_retries):
            try:
                return self.gpt(messages)
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                print(f"API call failed (attempt {attempt + 1}), retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
                self.retry_delay *= 2  # Exponential backoff
        
        raise Exception("Max retries exceeded")
    
    def safe_function_execution(self, function_call) -> Dict[str, Any]:
        """
        Execute function with comprehensive error handling.
        """
        try:
            return self.tools.execute_function_call(function_call)
        
        except json.JSONDecodeError as e:
            return {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": f"Error: Invalid JSON arguments - {str(e)}"
            }
        
        except KeyError as e:
            return {
                "type": "function_call_output", 
                "call_id": function_call.call_id,
                "output": f"Error: Missing required parameter - {str(e)}"
            }
        
        except Exception as e:
            return {
                "type": "function_call_output",
                "call_id": function_call.call_id, 
                "output": f"Error: Function execution failed - {str(e)}"
            }
```

#### Configuration and Customization

```python
class AssistantConfig:
    """Configuration class for customizing assistant behavior."""
    
    def __init__(self):
        self.model = 'gpt-4o-mini'
        self.max_iterations = 10
        self.max_function_calls_per_iteration = 5
        self.temperature = 0.1
        self.max_tokens = 1000
        self.search_result_limit = 5
        self.enable_debug_mode = False
        self.conversation_memory_limit = 50  # Max messages to keep in memory

def create_configured_assistant(config: AssistantConfig):
    """Create assistant with custom configuration."""
    
    # Configure tools based on config
    tools = Tools()
    faq_tools = CourseFAQTools(index)
    tools.add_tools_from_instance(faq_tools)
    
    # Create provider with config
    client_config = {
        'model': config.model,
        'temperature': config.temperature,
        'max_tokens': config.max_tokens
    }
    
    # Enhanced system prompt
    system_prompt = f"""
You're an intelligent course assistant with these capabilities:
- Search course FAQ (limit: {config.search_result_limit} results per search)
- Add new FAQ entries when requested  
- Provide general knowledge when appropriate

Configuration:
- Debug mode: {'enabled' if config.enable_debug_mode else 'disabled'}
- Max iterations: {config.max_iterations}
- Response style: Professional and helpful

Guidelines:
- Always prioritize accuracy over speed
- When uncertain, search for more information
- Provide sources for course-specific information
- Ask clarifying questions when needed
"""
    
    return ChatAssistant(
        tools=tools,
        developer_prompt=system_prompt,
        chat_interface=IPythonChatInterface(),
        client=client
    )
```

### Key Achievements in Part 3

1. **Modular Architecture**: Clean separation of concerns with dedicated classes
2. **Tool Management**: Sophisticated tools system with auto-registration capabilities  
3. **Error Resilience**: Comprehensive error handling and retry mechanisms
4. **Rich UI**: Enhanced display formatting for better user experience
5. **Configuration**: Flexible configuration system for different use cases
6. **Extensibility**: Easy addition of new tools and capabilities
7. **Production Readiness**: Code structured for real-world deployment

The evolution from basic RAG to this sophisticated agentic system demonstrates:
- **Architectural progression**: From simple functions to enterprise-ready classes
- **Feature sophistication**: From single searches to multi-step reasoning
- **User experience**: From plain text to rich, interactive interfaces
- **Maintainability**: From monolithic code to modular, testable components

This foundation prepares you to understand and work with production agent frameworks like **PydanticAI**, **OpenAI Agent SDK**, or build your own specialized agentic systems.

### Moving Beyond toy_AI_kit

While `toy_AI_kit` is excellent for learning, production systems often require:

1. **PydanticAI**: Type-safe tool definitions and validation
2. **LangChain/LangGraph**: Complex workflow orchestration  
3. **OpenAI Assistant API**: Hosted agent infrastructure
4. **Custom frameworks**: Domain-specific optimizations

The patterns and concepts you've learned here form the foundation for understanding and effectively using any of these advanced frameworks.

---

## Conclusion

This tutorial has taken you on a comprehensive journey from basic RAG systems to sophisticated agentic assistants. You've learned:

- **Foundational concepts**: RAG pipelines and agentic decision-making
- **Advanced techniques**: Function calling, iterative search, and tool composition
- **Production patterns**: Object-oriented design, error handling, and configuration

The agentic systems you can now build demonstrate true AI capabilities:
- **Autonomous decision-making** about when and how to use tools
- **Multi-step reasoning** through iterative search and analysis  
- **Tool composition** for complex task completion
- **Conversational memory** for natural multi-turn interactions

These skills prepare you to build production AI assistants, contribute to agent frameworks, and push the boundaries of what's possible with Large Language Models and agentic AI.

### Next Steps

1. **Experiment** with different tool combinations and prompting strategies
2. **Explore** production frameworks like PydanticAI and LangChain
3. **Build** domain-specific agents for your own use cases
4. **Study** the source code of open-source agent frameworks
5. **Contribute** to the growing ecosystem of agentic AI tools

The future of AI is agentic - and you're now equipped to be part of building it.

---

## Resources

- **Workshop Repository**: [https://github.com/alexeygrigorev/rag-agents-workshop](https://github.com/alexeygrigorev/rag-agents-workshop)
- **toy_AI_kit Library**: [https://github.com/alexeygrigorev/toyaikit](https://github.com/alexeygrigorev/toyaikit)
- **OpenAI Function Calling**: [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- **PydanticAI Framework**: [https://ai.pydantic.dev/](https://ai.pydantic.dev/)
- **Video Tutorials**: 
  - Part 1: [https://www.youtube.com/watch?v=GH3lrOsU3AU](https://www.youtube.com/watch?v=GH3lrOsU3AU)
  - Part 2: [https://www.youtube.com/watch?v=yS_hwnJusDk](https://www.youtube.com/watch?v=yS_hwnJusDk)

*Happy building! 🚀*