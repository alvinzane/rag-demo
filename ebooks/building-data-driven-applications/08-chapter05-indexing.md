# Chapter 5: Indexing with LlamaIndex

As this ebook edition doesn't have fixed pagination, the page numbers below are hyperlinked for reference only, based on the printed edition of this book.

This chapter provides an in-depth look at the different types of indexes available in LlamaIndex. It explains how indexes work, and their key capabilities, customization options, underlying architectures, and use cases. Overall, this chapter serves as a guide for leveraging the indexing functionality within LlamaIndex to build performant and scalable RAG systems. Let's get started!

Throughout this chapter, we'll cover the following topics:

- Indexing data – a bird's-eye view
- Understanding the Vector Store Index
- Understanding embeddings
- Persisting and re-using Indexes
- Exploring other Index types in LlamaIndex
- Building Indexes on top of other Indexes with ComposableGraph
- Estimating the potential cost of building and querying Indexes

## Technical requirements

For this chapter, you will need to install the following package in your environment:

- *ChromaDB*: https://www.trychroma.com/

In addition, there are two integration packages that will be required by the sample code:

- *Chroma Vector Store*: https://pypi.org/project/llama-index-vector-stores-chroma/
- *Hugging Face embeddings*: https://pypi.org/project/llama-index-embeddings-huggingface/

All code samples from this chapter can be found in the **ch5** subfolder of the book's GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex

## Indexing data – a bird's-eye view

We briefly discussed the importance and general functioning of Indexes in a RAG application in *Chapter 3*, in the section titled *Uncovering the essential building blocks of LlamaIndex – documents, nodes, indexes*. Now, it is time to have a closer look at the different indexing methods available in LlamaIndex with their advantages, disadvantages, and specific use cases.

In principle, data can be accessed even without an Index. But it's like reading a book without a table of contents. As long as it's about a story that has continuity and can be read sequentially, section by section, and chapter by chapter, reading will be a pleasure. However, things change when we need to quickly search for a specific topic in that book. Without a table of contents, the search process will be slow and cumbersome.

In LlamaIndex, however, **Indexes** represent more than just a simple table of contents. An Index provides not only the necessary structure for navigation but also the concrete mechanisms to update or access it. That includes the logic for the **retrievers** and the mechanisms used for fetching data, which we will discuss in detail during *Chapter 6*, *Querying Our Data, Part 1 – Context Retrieval*.

In this book, I've kept things simple, giving you the basics of how Indexes work and providing some examples to help you understand their usage. Exploring every possible way to use and mix these Indexes would be a huge task and that's not what we're about here.

We'll talk later about what makes each type of Index unique, but first, let's see what they all have in common.

### Common features of all Index types

Each type of Index in LlamaIndex has its own characteristics and functions, but because all of them inherit the **BaseIndex** class, there are certain features and parameters they share, which can be customized for any kind of Index:

- **Nodes**: All Indexes are based on nodes and we can choose which Nodes are included in the Index. Plus, all Index types provide methods to insert new Nodes or delete existing ones, allowing for dynamic updates to the Index as your data changes. We can either build an Index from pre-existing Nodes by providing the Nodes directly to the Index constructor like this **vector_index = VectorStoreIndex(nodes)** or we can provide a list of documents as an input using **from_documents()** and let the Index extract the Nodes by itself. Keep in mind that we can use **Settings** – before actually building the Index – to customize its underlying mechanics. As we discussed during *Chapter 3* in the *Understanding how settings can be used for customization* section, this simple class allows for different settings such as changing the LLM, embedding model, or default Node parser used by an Index.

- **The storage context**: The storage context defines how and where the data (documents and nodes) for the Index is stored. This customization is crucial for managing data storage efficiently, depending on the application's requirements.

- **Progress display**: The **show_progress** option lets us choose whether to display progress bars during long-running operations such as building the Index. Implemented using the **tqdm** Python library, this feature can be useful for monitoring the progress of large indexing tasks.

- **Different retrieval modes**: Each Index allows for different pre-defined retrieval modes, which can be set to match the specific needs of your application. And you can also customize or extend the Retriever classes to change how queries are processed and how results are retrieved from the Index. More on that during *Chapter 6*, *Querying Our Data, Part 1 – Context Retrieval*.

- **Asynchronous operations**: The **use_async** parameter implemented by some of the Indexes determines whether certain operations should be performed asynchronously. Asynchronous processing allows the system to manage multiple operations concurrently, rather than waiting for each operation to be completed sequentially. This can be important for performance optimization, especially when dealing with large datasets or complex operations.

### Quick note

An important thing to consider before diving further and starting to tinker with the sample code is that indexing often relies on LLM calls for summarizing or embedding purposes. Just like in the case of metadata extraction, which we covered in *Chapter 4*, *Ingesting Data into Our RAG Workflow*, indexing in LlamaIndex may also raise cost and privacy concerns. Make sure you read the cost-related section at the end of this chapter before running any large-scale experiments to test your ideas.

Let's start with our first and most frequently used Index type.

## Understanding the VectorStoreIndex

As this ebook edition doesn't have fixed pagination, the page numbers below are hyperlinked for reference only, based on the printed edition of this book.

In LlamaIndex, the **VectorStoreIndex** stands out as the workhorse, being the most commonly utilized type of Index.

For most RAG applications, a **VectorStoreIndex** might be the best solution because it facilitates the construction of Indexes on collections of Documents where **embeddings** for the input text chunks are stored within the **Vector Store** of the Index. Once constructed, this Index can be used for efficient querying because it allows for **similarity searches** over the embedded representations of the text, making it highly suitable for applications requiring fast retrieval of relevant information from a large collection of data. Don't worry if you're not yet familiar with terms such as embeddings, vector store, or similarity searching, because we'll cover them in the following sections. The **VectorStoreIndex** class in LlamaIndex supports these operations by default and also allows for asynchronous calls and progress tracking, which can improve performance and user experience in typical RAG scenarios.

### A simple usage example for the VectorStoreIndex

Here's the most basic way of constructing a **VectorStoreIndex**:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
print("Index created successfully!")
```

As you can see, in just a few lines of code, we have ingested the Documents and the **VectorStoreIndex** took care of everything. Note that using this approach, we have skipped the Node parsing step entirely, because the Index did that by itself by using the **from_documents()** method.

There are several parameters that we can customize for the **VectorStoreIndex**:

- **use_async**: This parameter enables asynchronous calls. By default, it is set to **False**.
- **show_progress**: This parameter shows progress bars during Index construction. The default value is **False**.
- **store_nodes_override**: This parameter forces LlamaIndex to store Node objects in the Index store and document store, even if the vector store keeps text. This can be useful in scenarios where you need direct access to Node objects, even if their content is already stored in the vector store. We'll talk in more detail about the Index store, document store, and vector store later in this chapter. The default setting for this parameter is **False**.

Let's have a look at *Figure 5.1* for a visual representation of this type of Index:

![Figure 5.1 – The structure of a VectorStoreIndex](images/chapter-05-figure-05-1-structure-vectorstoreindex.jpg)

*Figure 5.1 – The structure of a VectorStoreIndex*

The **VectorStoreIndex** took the ingested Documents, breaking them down into Nodes. It used the default parameters for text splitter, chunk size, chunk overlap, and so on. Of course, we could have customized all these parameters if we wanted to.

### Note

**Fixed-size chunking** simply splits text into same-sized chunks, optionally with some overlap. Although computationally cheap and simple to implement, this simple chunking may not always be the best approach. Performance testing various chunk sizes is key to optimizing for an application's particular needs.

The nodes containing chunks of the original text were then *embedded* into a high-dimensional vector space using a language model. The embedded vectors were stored within the vector store component of the Index. Next, when a query is made, the query text will be similarly embedded and compared against the stored vectors using a **similarity measure** identified with a method called **cosine similarity**. The most similar vectors – and thus the most relevant document chunks – will be returned as the query result. This process enables rapid, semantically aware retrieval of information, leveraging the mathematical properties of vector spaces to find the documents that best answer the user's query.

Sounds a bit confusing? Let's go through these concepts together in the next section.

## Understanding embeddings

In simple terms, **vector embeddings** represent a machine-understandable data format. They capture meaning and may conceptually represent a word, an entire document, or even non-textual information such as images and sounds. In a way, embeddings represent a standard language of thought for an LLM. In the context of an LLM, they serve as a foundational representation through which the model understands and processes information. They transform diverse and complex data into a uniform, high-dimensional space where the LLM can perform operations such as comparison, association, and prediction more effectively. *Figure 5.2* provides an illustration of the process of embedding data:

![Figure 5.2 – How an embedding model converts data into numerical representations](images/chapter-05-figure-05-2-embedding-model-converts-data.jpg)

*Figure 5.2 – How an embedding model converts data into numerical representations*

Because it's all about math under the hood. And math works well with numbers – more precisely, large lists of floating-point numbers, where each number represents a dimension in a hypothetical vector space. The LLM can work with these arrays of numbers to understand, interpret, and generate responses based on the input it receives. Essentially, these numbers in the vector embeddings allow the LLM to *see* and *think* about the data in a way that's meaningful and structured.

The beauty of this system lies in its ability to handle ambiguity and complexity. The model can understand semantic relationships between words, such as synonyms, antonyms, and more complex linguistic patterns. In the case of polysemous words, the same word can have different meanings in different contexts. For example, the word *bank* can refer to the side of a river or a financial institution. Vector embeddings help the LLM understand these nuances by providing context-sensitive representations. So, in one situation, *bank* might be closely associated with words such as *river* and *shore*, while in another, it's more closely linked to *money* and *account*.

### Quick note

An important factor to consider is that the size of text chunks being embedded impacts precision – too small and context is lost; too large and all that additional detail may dilute the meaning.

In case you're not very familiar with embeddings yet, the following example could be useful to get a better grasp of the concept. Let's assign some *arbitrary* vector embeddings to three randomly chosen sentences:

- **Sentence 1**: The quick brown fox jumps over the lazy dog
- **Sentence 2**: A fast dark-colored fox leaps above a sleepy canine
- **Sentence 3**: Apples are sweet and crunchy

In a real-world scenario, the embeddings associated with each of these sentences would be calculated automatically by using an **embedding model**. This is a specialized artificial intelligence model used to convert complex data such as text, images, or graphs into a numerical format. The embeddings would also normally be high-dimensional, but for the sake of explanation, I'll use simple, three-dimensional, arbitrarily chosen vectors. Here are the hypothetical embeddings for the three sentences:

- **Sentence 1 Embedding**: [0.8, 0.1, 0.3]
- **Sentence 2 Embedding**: [0.79, 0.14, 0.32]
- **Sentence 3 Embedding**: [0.2, 0.9, 0.5]

These numbers are purely conceptual and are meant to show that sentences 1 and 2, which have similar meanings, have embeddings that are closer to each other in vector space. *Sentence 3*, which has a different meaning, has an embedding that is farther away from the first two. Have a look at *Figure 5.3* for a straightforward visual comparison of the three embeddings:

![Figure 5.3 – A comparison of the three embedded sentences in a 3D space](images/chapter-05-figure-05-3-comparison-embedded-sentences-3d.jpg)

*Figure 5.3 – A comparison of the three embedded sentences in a 3D space*

When we visualize them in a three-dimensional space, sentences 1 and 2 are plotted near each other, while sentence 3 will be plotted at a distance. This spatial representation is what allows machine learning models to determine semantic similarity.

When you search using a query on a vector store Index in order to retrieve useful context, LlamaIndex converts your search terms into a similar embedding and then finds the closest matches among the pre-computed embeddings of your text chunks.

We call this process **similarity** or **distance search**. So, when you encounter the term **top-k similarity search**, you should know that it relies on an algorithm that calculates the similarity between vector embeddings. It takes a vector embedding as an input and returns the most similar *k* number of vectors found in the vector store. Because the initial vector and the *top-k* returned neighbors are similar to each other, we can consider their meanings to be conceptually similar. Now you understand why I have previously called embeddings a *standard language of thought* for an LLM. It doesn't really matter anymore whether they represent text, images, or any other types of information. We measure their similarity in numbers.

The only thing that may be implemented differently, depending on our use case, is the actual formula for defining that distance or similarity.

Spoiler alert: a bit of mathematical concepts up next.

### Understanding similarity search

In the realms of machine learning and deep learning, the concept of similarity search is very important. It forms the backbone of many applications, from recommendation systems and information retrieval to clustering and classification tasks. As models and systems interact with high-dimensional data, identifying patterns and relationships between data points becomes essential. This involves measuring how *close* or *similar* data elements are, a task that often takes place in a vector space where each item is represented as a vector.

Locating points in this space that are near each other enables machines to assess similarity and, by extension, to make decisions, draw inferences, or, in our case, retrieve information based on that assessment of closeness. With the advent of embeddings in deep learning, the need for effective similarity search has grown. As embeddings capture the semantic meaning of the data they represent, performing similarity searches on these vectors allows machines to understand content at a level approaching human cognition.

Let's explore the methods that LlamaIndex currently employs to measure the similarity between vectors, each with its unique advantages and applicability.

#### Cosine similarity

This method measures the cosine of *the angle* between two vectors. Imagine two arrows pointing in different directions; the smaller the angle between them, the more similar they are.

Have a look at *Figure 5.4*, which depicts a cosine similarity comparison between two vectors:

![Figure 5.4 – How a cosine similarity comparison would look](images/chapter-05-figure-05-4-cosine-similarity-comparison.jpg)

*Figure 5.4 – How a cosine similarity comparison would look*

In terms of embeddings, a small angle (or a high cosine similarity score, close to 1) indicates that the content they represent is similar. This method is particularly useful in text analysis because it is less affected by the length of the documents and focuses more on their direction or orientation in the vector space.

### Note

Cosine similarity is also the default method used by LlamaIndex for calculating similarity between embeddings.

#### Dot product

Also called the **scalar product**, because it is represented by a single value, this is another method of calculating how well two vectors align with each other. To calculate the scalar product of two vectors, the algorithm multiplies the corresponding elements of the vectors and then sums these products.

Let's take a simple example of *vector A*: [2,3] and *vector B*: [4,1]. The **dot product** is calculated by multiplying their corresponding elements: (2×4) + (3×1), which gives us 8 + 3 = 11. Thus, the dot product of these two vectors is 11.

*Figure 5.5* exemplifies this concept:

![Figure 5.5 – Calculating similarity using the dot product method](images/chapter-05-figure-05-5-calculating-similarity-dot-product.jpg)

*Figure 5.5 – Calculating similarity using the dot product method*

In the preceding diagram, the dot product is visualized by projecting one vector onto the other. This projection illustrates the geometric interpretation of the dot product. It's calculated by projecting the components of one vector in the direction of the other and then multiplying these projected components by the corresponding components of the second vector. The sum of these products gives us the dot product. This visualization helps us understand that the dot product is not just a measure of how vectors point in the same direction; it also incorporates their lengths.

Higher values of the dot product mean higher similarities between vectors. In contrast with the cosine method, the dot product is sensitive both to the length of the two vectors compared and their relative direction. Unlike the dot product, cosine similarity normalizes the dot product by the magnitudes of the vectors. This normalization makes cosine similarity solely a measure of the directional alignment between vectors, independent of their lengths.

#### Euclidean distance

This method calculates the straight-line distance between two points in space. In the context of embeddings, it measures how far apart two vectors are in the vector space.

*Figure 5.6* shows how Euclidean distance is calculated:

![Figure 5.6 – Calculating similarity using Euclidean distance](images/chapter-05-figure-05-6-euclidean-distance-two-vectors.jpg)

*Figure 5.6 – Calculating similarity using Euclidean distance*

Unlike cosine similarity and dot product, where higher values indicate greater similarity, with Euclidean distance, smaller values indicate greater similarity. This is because Euclidean distance measures the actual distance between points – the closer they are, the more similar they are considered to be.

The choice of similarity metric can significantly impact the performance of your RAG system, depending on the nature of your data and the specific requirements of your application.

### Customizing the embedding model

By default, LlamaIndex uses OpenAI's text-embedding-ada-002 model for generating embeddings. However, you can customize this to use different embedding models based on your needs:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use a Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
```

Different embedding models have different characteristics:
- **OpenAI models**: High quality, but require API calls and incur costs
- **Hugging Face models**: Can run locally, free to use, but may require more computational resources
- **Sentence Transformers**: Optimized for semantic similarity tasks

### Working with vector stores

The **VectorStoreIndex** can work with different vector store backends. By default, it uses an in-memory vector store, but you can configure it to use persistent vector databases:

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.create_collection("my_collection")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index with custom vector store
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

Popular vector store options include:
- **Chroma**: Open-source, easy to use
- **Pinecone**: Managed service, highly scalable
- **Weaviate**: Open-source with advanced features
- **Qdrant**: High-performance vector database

## Persisting and reusing Indexes

One of the key features of LlamaIndex is the ability to persist Indexes to disk and reload them later. This is crucial for production applications where you don't want to rebuild indexes every time your application starts.

### Basic persistence

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, StorageContext

# Create and persist index
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")

# Later, load the index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### Storage context components

The storage context consists of three main components:

1. **Document Store**: Stores the original documents
2. **Index Store**: Stores index-specific metadata
3. **Vector Store**: Stores the vector embeddings

```python
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.simple import SimpleVectorStore

# Create custom storage context
docstore = SimpleDocumentStore()
index_store = SimpleIndexStore()
vector_store = SimpleVectorStore()

storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=vector_store
)
```

### Managing multiple indexes

When working with multiple indexes, you need to assign unique IDs:

```python
# Create multiple indexes with IDs
vector_index = VectorStoreIndex.from_documents(documents)
vector_index.set_index_id("vector_index")

summary_index = SummaryIndex.from_documents(documents)
summary_index.set_index_id("summary_index")

# Persist both indexes
storage_context = StorageContext.from_defaults()
storage_context.persist(persist_dir="./storage")

# Load specific index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
vector_index = load_index_from_storage(storage_context, index_id="vector_index")

## Exploring other index types in LlamaIndex

While the **VectorStoreIndex** is the most commonly used, LlamaIndex provides several other index types, each optimized for different use cases and query patterns.

### SummaryIndex

The **SummaryIndex** is the simplest type of index. It stores nodes in a sequential list and performs a linear scan through all nodes during querying. This makes it suitable for small datasets or when you need to ensure all relevant information is considered.

```python
from llama_index.core import SummaryIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("files").load_data()
index = SummaryIndex.from_documents(documents)

# Query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

**Use cases for SummaryIndex:**
- Small document collections
- When comprehensive coverage is more important than speed
- Summarization tasks where all content should be considered

### TreeIndex

The **TreeIndex** builds a hierarchical tree structure from the bottom up. It creates parent nodes that summarize their children, forming a tree where leaf nodes contain the original text and internal nodes contain summaries.

```python
from llama_index.core import TreeIndex

documents = SimpleDirectoryReader("files").load_data()
index = TreeIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What are the key points?")
```

**Use cases for TreeIndex:**
- Large documents that benefit from hierarchical organization
- When you need both detailed and high-level information
- Documents with clear hierarchical structure

### KeywordTableIndex

The **KeywordTableIndex** extracts keywords from each node and creates a mapping from keywords to nodes. During querying, it matches query keywords to find relevant nodes.

```python
from llama_index.core import KeywordTableIndex

documents = SimpleDirectoryReader("files").load_data()
index = KeywordTableIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("machine learning algorithms")
```

**Use cases for KeywordTableIndex:**
- When exact keyword matching is important
- Technical documents with specific terminology
- When semantic similarity might miss important exact matches

### KnowledgeGraphIndex

The **KnowledgeGraphIndex** extracts entities and relationships from text to build a knowledge graph. It's particularly useful for understanding connections between different concepts.

```python
from llama_index.core import KnowledgeGraphIndex

documents = SimpleDirectoryReader("files").load_data()
index = KnowledgeGraphIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("How are these concepts related?")
```

**Use cases for KnowledgeGraphIndex:**
- Documents with many interconnected concepts
- When relationship understanding is crucial
- Research papers and technical documentation

### Choosing the right index type

The choice of index type depends on several factors:

1. **Data size**: SummaryIndex for small datasets, VectorStoreIndex for large ones
2. **Query patterns**: KeywordTableIndex for exact matches, VectorStoreIndex for semantic search
3. **Data structure**: TreeIndex for hierarchical data, KnowledgeGraphIndex for interconnected concepts
4. **Performance requirements**: VectorStoreIndex for fast retrieval, SummaryIndex for comprehensive coverage

## Building Indexes on top of other Indexes with ComposableGraph

LlamaIndex allows you to compose multiple indexes together using **ComposableGraph**. This enables sophisticated multi-level indexing strategies.

### Basic composition

```python
from llama_index.core import ComposableGraph, VectorStoreIndex, SummaryIndex

# Create individual indexes
vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)

# Create a graph that combines both indexes
graph = ComposableGraph.from_indices(
    SummaryIndex,  # Top-level index type
    [vector_index, summary_index],  # Child indexes
    index_summaries=["Vector-based search", "Comprehensive summary"]
)

query_engine = graph.as_query_engine()
response = query_engine.query("What is the main topic?")
```

### Advanced composition strategies

You can create more complex compositions:

```python
# Create domain-specific indexes
tech_docs = SimpleDirectoryReader("tech_docs").load_data()
business_docs = SimpleDirectoryReader("business_docs").load_data()

tech_index = VectorStoreIndex.from_documents(tech_docs)
business_index = VectorStoreIndex.from_documents(business_docs)

# Compose them with a routing layer
graph = ComposableGraph.from_indices(
    TreeIndex,
    [tech_index, business_index],
    index_summaries=[
        "Technical documentation and API references",
        "Business processes and policies"
    ]
)
```

**Benefits of ComposableGraph:**
- Combines strengths of different index types
- Enables domain-specific routing
- Provides fallback mechanisms
- Allows for sophisticated query strategies

## Estimating the potential cost of building and querying Indexes

Understanding the cost implications of different indexing strategies is crucial for production deployments. Costs come from two main sources: embedding generation and LLM calls.

### Cost factors

1. **Embedding costs**: Based on the number of tokens processed
2. **LLM costs**: For summarization and query processing
3. **Storage costs**: For persisting indexes and vector stores
4. **Compute costs**: For similarity calculations and processing

### Estimating embedding costs

```python
import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings

# Set up token counting
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
callback_manager = CallbackManager([token_counter])
Settings.callback_manager = callback_manager

# Build index and track tokens
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)

print(f"Embedding tokens used: {token_counter.total_embedding_token_count}")
print(f"LLM tokens used: {token_counter.total_llm_token_count}")

# Estimate costs (example rates)
embedding_cost = (token_counter.total_embedding_token_count / 1000) * 0.0001  # $0.0001 per 1K tokens
llm_cost = (token_counter.total_llm_token_count / 1000) * 0.002  # $0.002 per 1K tokens

print(f"Estimated embedding cost: ${embedding_cost:.4f}")
print(f"Estimated LLM cost: ${llm_cost:.4f}")
```

### Cost optimization strategies

1. **Use local embedding models** to eliminate API costs:
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.embed_model = embed_model
```

2. **Optimize chunk sizes** to balance quality and cost:
```python
from llama_index.core.node_parser import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=512,  # Smaller chunks = more embeddings but better precision
    chunk_overlap=50
)
```

3. **Use caching** to avoid reprocessing:
```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

cache = IngestionCache()
pipeline = IngestionPipeline(
    transformations=[text_splitter, embed_model],
    cache=cache
)
```

4. **Choose appropriate index types** based on cost-performance trade-offs:
- **SummaryIndex**: Low build cost, high query cost
- **VectorStoreIndex**: High build cost, low query cost
- **KeywordTableIndex**: Medium build cost, medium query cost

## Indexing our PITS study materials – hands-on

With a solid understanding of how indexing works in LlamaIndex, we're now ready to implement the indexing logic in our tutoring application.

Let's create the **index_builder.py** module. This module takes care of Index creation. In the current implementation, it creates two Indexes: a **VectorStoreIndex** and a **TreeIndex**. As you can see, this is a very basic implementation and there is definitely room for improvement. Let's handle the imports first:

```python
from llama_index.core import (
    VectorStoreIndex,
    TreeIndex,
    load_index_from_storage
)
from llama_index.core import StorageContext
from global_settings import INDEX_STORAGE
from document_uploader import ingest_documents
```

Next, we'll implement our Index building function:

```python
def build_indexes(nodes):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=INDEX_STORAGE
        )
        vector_index = load_index_from_storage(
            storage_context,
            index_id="vector"
        )
        tree_index = load_index_from_storage(
            storage_context,
            index_id="tree"
        )
        print("All indices loaded from storage.")
```

We first check to see whether the Indexes have already been persisted to disk. If affirmative, then we leverage persistence to avoid the additional cost of rebuilding them.

### Note on: Notice the usage of index_id

Because we have persisted more than one Index in the same storage folder – **INDEX_STORAGE** – when using **load_index_from_storage**, we need to specify their individual IDs so that LlamaIndex can identify the correct Index.

If we cannot find them in the **INDEX_STORAGE** folder, we proceed to build them from the nodes. We also set an ID for each Index using **set_index_id** so that we can load them correctly in future sessions:

```python
    except Exception as e:
        print(f"Error occurred while loading indices: {e}")
        storage_context = StorageContext.from_defaults()

        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context
        )
        vector_index.set_index_id("vector")

        tree_index = TreeIndex(
            nodes,
            storage_context=storage_context
        )
        tree_index.set_index_id("tree")

        storage_context.persist(
            persist_dir=INDEX_STORAGE
        )
        print("New indexes created and persisted.")

    return vector_index, tree_index
```

The **build_indexes** function returns the two Index objects that we'll use later in our application.

That's it for now. We'll take the next steps during *Chapter 6*, *Querying Our Data, Part 1 – Context Retrieval*.

## Summary

In this chapter, we explored various indexing strategies and architectures within LlamaIndex. Indexes provide essential capabilities for building performant RAG systems.

Throughout the chapter, we looked at the **VectorStoreIndex**, which is the most commonly used Index type. We also gained an understanding of embeddings, vector stores, similarity search, and storage contexts. These are key concepts related to the **VectorStoreIndex**.

We also covered other Index types such as **SummaryIndex** for simple linear scans, **KeywordTableIndex** for keyword search, **TreeIndex** for hierarchical data, and **KnowledgeGraphIndex** for relationship-based queries. **ComposableGraph** was introduced as a tool for building multi-level Indexes, and cost estimation techniques were discussed together with best practices.

Overall, this chapter provided an overview of indexing capabilities in LlamaIndex, laying the foundation for building sophisticated and efficient RAG applications.

See you in *Chapter 6*, where we'll discuss methods for querying our data in LlamaIndex.
```
