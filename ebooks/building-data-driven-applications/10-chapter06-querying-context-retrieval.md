# Chapter 6: Querying Our Data, Part 1 – Context Retrieval

The focus of this chapter will be on understanding the querying capabilities of LlamaIndex in an RAG workflow. We'll be covering the overall working of the querying system, mostly focusing on the retrieval capabilities of the framework.

Here are the main sections that will be covered in this chapter:

- Learning about query mechanics – an overview
- Understanding the basic retrievers
- Building more advanced retrieval mechanisms
- Increasing efficiency with asynchronous retrieval
- Working with metadata filters, tools, and selectors
- Transforming queries and generating sub-queries
- Understanding the concepts of dense and sparse retrieval

## Technical requirements

For this chapter, you will need to install the **Rank-BM25** package in your environment. You can find it at https://pypi.org/project/rank-bm25/.

Two additional integration packages are required to run the sample code:

- **OpenAI Question Generator**: https://pypi.org/project/llama-index-question-gen-openai/
- **BM25 Retriever**: https://pypi.org/project/llama-index-retrievers-bm25/
- All the code samples for this chapter can be found in the ch6 subfolder of this book's GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

## Learning about query mechanics – an overview

In this chapter, we will finally begin to reap the fruits of our work so far. Document ingestion, parsing and segmenting, metadata extraction, and index building were all just preparatory steps for what we are about to discuss: **querying**. At the heart of any RAG workflow is the idea of being able to bring relevant context into the prompt we use in the LLM query. So far, we have been concerned with constructing and organizing this context, but now, it is time to use it and extract the best possible answers from our interactions with LLMs. In the following sections, we will discuss various techniques that LlamaIndex provides us for the query part. As usual, we will start with the simplest query methods – called *naive* methods in jargon – and then discuss more advanced query variants.

First, we need to understand the typical steps in the query process: **retrieval**, **postprocessing**, and **response synthesis**.

In *Chapter 3*, *Kickstarting Your Journey with LlamaIndex*, in the *Indexes* section, we discussed the simplest way to go through the three steps – using **QueryEngine** but built very simply by running **index.as_query_engine()**. This is very simple but not necessarily always effective as this *naive* way of querying an index is just the tip of the iceberg. We will now explore the three mechanisms individually and understand how they work and the customizable options they offer.

First, we'll focus on **retrievers**.

## Understanding the basic retrievers

**Retrieval mechanisms** are a central element in any RAG system. Although they work in different ways, all types of retrievers are based on the same principle: they browse an index and select the relevant nodes to build the necessary context. Each index type offers several retrieval modes, each providing different features and customization options. Regardless of the retriever type, the result that will be returned is in the form of a **NodeWithScore** object – a structure that combines a node with an associated score. The score can be useful further in the RAG flow because it allows us to sort the returned nodes according to their relevance. However, keep in mind that while all retrievers return **NodeWithScore**, not all of them associate a specific node score.

As usual, LlamaIndex offers multiple alternatives to accomplish a task, so a retriever can be constructed in several ways. The simplest path is direct construction from an **Index** object. Assuming that we have already dealt with document ingestion, the following code builds an index and then builds a retriever based on the structure of the index:

```python
from llama_index.core import SummaryIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
retriever = summary_index.as_retriever(
    retriever_mode='embedding'
)
result = retriever.retrieve("Tell me about ancient Rome")
print(result[0].text)
```

In the previous example, the generated retriever is of the **SummaryIndexRetriever** type. This is the default retriever for this index.

The second option is direct instantiation, as shown in the following example:

```python
from llama_index.core import SummaryIndex, SimpleDirectoryReader
from llama_index.core.retrievers import SummaryIndexEmbeddingRetriever
documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
retriever = SummaryIndexEmbeddingRetriever(
    index=summary_index
)
result = retriever.retrieve("Tell me about ancient Rome")
print(result[0].text)
```

In the next section, we'll go through a list of retrieval options that are available for each index type. Next to each retriever type, I've specified how it can be instantiated from the corresponding index. I warn you now that a lot of information has been condensed in the next section. However, it is useful information that you can bookmark and come back to later when you start building real applications with the LlamaIndex framework.

So, here's the list of retrievers for each type of index.

## The VectorStoreIndex retrievers

We have two retriever options available for this index. Let's have a look at how they work and how to customize them for different use cases.

### VectorIndexRetriever

The default retriever that's used by **VectorStoreIndex** is **VectorIndexRetriever**. It can easily be constructed using the following command:

```python
VectorStoreIndex.as_retriever()
```

As expected, since **VectorStoreIndex** is one of the most sophisticated and widely used indexes, this retriever is also complex.

*Figure 6.1* exemplifies its operating mode:

![Figure 6.1 – Node retrieval using VectorIndexRetriever](images/chapter-06-figure-06-1-node-retrieval-vectorindexretriever.jpg)

*Figure 6.1 – Node retrieval using VectorIndexRetriever*

This retriever operates by converting queries into vectors and then performing *similarity-based* searches in the vector space. Several parameters can be customized for different use cases:

- **similarity_top_k**: This defines the number of *top (k)* results returned by the retriever. This determines how many of the most similar results are returned for each query. For example, if we want a broader search, we can change the default value, which is **2**.

- **vector_store_query_mode**: This sets the query mode of the vector store. Different variants of external vector stores, such as *Pinecone* (https://www.pinecone.io/), *OpenSearch* (https://opensearch.org/), and others, support different query modes. This is the mechanism by which we can make best use of their search capabilities.

- **filters**: Remember that in *Chapter 3*, in the *Nodes* section, we saw how to add metadata to our nodes? Well, we can use this metadata to narrow down the search scope of the retriever. We will see a practical example of this in this chapter, where we will use metadata filters to implement a simple system for filtering nodes returned by an index.

- **alpha**: This one is useful when using a hybrid search mode (a combination of sparse and **dense search**). We will discuss the difference between sparse and dense search in more detail later in this chapter.

- **sparse_top_k**: The number of top results for the **sparse search**. This is relevant in hybrid search modes. The previous mention applies here also.

- **doc_ids**: Similar to metadata filters, but slightly coarser, **doc_ids** can be used to restrict the search to a specific subset of documents. For example, suppose the organization uses a common knowledge base that is shared by all departments. At the same time, however, the organization has a clear naming convention for documents. If the department's name or code is found in the document name, we could use this parameter to limit a user's query to documents in their department only.

- **node_ids**: This parameter is similar to **doc_ids** but refers to node IDs within the index. This can give us even more granular control over the information that's returned by the retriever.

- **vector_store_kwargs**: This parameter can pass additional arguments that are specific to each vector store so that they can be sent at query time.

As a secure design principle, security should be implemented as early as possible in the life cycle of an application. This is also true for an RAG application. For example, if we want to better control access to information, we should filter the information that's processed by the application as early as possible. In an RAG flow, which means from the moment it is retrieved – if not earlier. There are ways to filter the information later in the query engine – for example, in post-processing or even in response synthesis – but it is much easier not to introduce risks in the first place by introducing information into the flow that is outside the user's security context. There is also a cost issue. Since much of the processing in an RAG flow is based on LLM ingestion, the less information we process, the lower the cost.

### VectorIndexAutoRetriever

All the parameters we discussed earlier regarding **VectorIndexRetriever** are very useful when we know exactly what we are looking for and understand the structure of the data very well. Unfortunately, in some situations, we will be dealing with complex structures or ambiguities in the indexed data.

**VectorIndexAutoRetriever** is a more advanced form of retriever that can use an LLM to automatically set query parameters in a vector store based on a natural language description of the content and supporting metadata. This is particularly useful when users are unfamiliar with the structure of the data or do not know how to formulate an effective query. In these situations, this retriever can transform vague or unclear queries into more structured queries and better leverage the capabilities of the vector store, thus increasing the chances of finding relevant results. Since a detailed discussion of this mechanism would take several pages and I am probably digressing too much from the main topic, if you want to learn more about how it works, I suggest that you consult the official documentation at https://docs.llamaindex.ai/en/stable/examples/vector_stores/elasticsearch_auto_retriever.html.

### The SummaryIndex retrievers

There are three retriever options available for this index. Let's take a look.

### SummaryIndexRetriever

This retriever can be built using the following command:

```python
SummaryIndex.as_retriever(retriever_mode = 'default')
```

This is the default retriever for **SummaryIndex**. As seen in *Figure 6.2*, it has a very simple approach – it returns all nodes in the index without applying any filtering or sorting:

![Figure 6.2 – Retrieving nodes using SummaryIndexRetriever](images/chapter-06-figure-06-2-retrieving-nodes-summaryindexretriever.jpg)

*Figure 6.2 – Retrieving nodes using SummaryIndexRetriever*

This is useful when we want to get a complete view of the data in the index, without having to filter or sort the results. No relevance score is returned for the nodes.

### SummaryIndexEmbeddingRetriever

We can build this one with the following command:

```python
SummaryIndex.as_retriever(retriever_mode='embedding')
```

This retriever relies on embeddings to retrieve nodes from **SummaryIndex**. While **SummaryIndex** itself stores nodes in plain text, this retriever uses an embedding model to convert these plain text nodes into embeddings when a query is made. Have a look at *Figure 6.3* to get a better view of its operating mode:

![Figure 6.3 – Inner workings of SummaryIndexEmbeddingRetriever](images/chapter-06-figure-06-3-inner-workings-summaryindexembeddingretriever.jpg)

*Figure 6.3 – Inner workings of SummaryIndexEmbeddingRetriever*

The embeddings are created dynamically as needed for retrieval, rather than being stored persistently with the index. The **similarity_top_k** parameter determines the number of nodes to return, based on their similarity to the query. This retriever is useful for finding the most relevant nodes concerning a given query by using similarity computation.

For each selected node, the retriever calculates a similarity score – based on embeddings – which is then returned alongside the node as **NodeWithScore**. This score is a reflection of the extent to which each node corresponds to the query.

### SummaryIndexLLMRetriever

This retriever can be built using the following command:

```python
SummaryIndex.as_retriever(retriever_mode='llm')
```

As its name suggests, this retriever uses an LLM to retrieve nodes from **SummaryIndex**. It uses a prompt to select the most relevant nodes. Check out *Figure 6.4* for an overview of its approach:

![Figure 6.4 – SummaryIndexLLMRetriever in action](images/chapter-06-figure-06-4-summaryindexllmretriever-in-action.jpg)

*Figure 6.4 – SummaryIndexLLMRetriever in action*

If we wish, we can override the default prompt using the **choice_select_prompt** parameter. Queries are processed in batches; the size of each batch is determined by the **choice_batch_size** parameter. Optionally, we can also provide the **format_node_batch_fn** and **parse_choice_select_answer_fn** functions as parameters. These are used to format the batch of nodes and parse the LLM responses. The **parse_choice_select_answer_fn** function is also responsible for calculating node-specific relevance scores. The scores are determined by parsing the LLM responses. These scores are then associated with the corresponding nodes and returned as **NodeWithScore**. If we don't want to use the default LLM, that's not a problem: the retriever accepts **service_context** as a parameter. In Chapter 3, we saw how to customize the default LLM using **ServiceContext**.

This type of retriever is useful in complex search systems where LLMs can provide contextual and detailed answers to queries.

Next, we'll talk about retrievers for **DocumentSummaryIndex**.

## The DocumentSummaryIndex retrievers

For this index, we only have two retrieval options. Let's take a look.

### DocumentSummaryIndexLLMRetriever

We can build this with the following command:

```python
DocumentSummaryIndex.as_retriever(retriever_mode='llm')
```

This retriever uses an LLM to select relevant summaries from an index of document summaries. You can get a better understanding of how it works by looking at *Figure 6.5*:

![Figure 6.5 – How DocumentSummaryIndexLLMRetriever works](images/chapter-06-figure-06-5-how-documentsummaryindexllmretriever-works.jpg)

*Figure 6.5 – How DocumentSummaryIndexLLMRetriever works*

This retriever processes queries in batches, with each batch containing a specified number of nodes to send to the LLM for evaluation. The **choice_batch_size** parameter can be used to specify the size of a batch. The retriever can use a custom prompt provided via the **choice_select_prompt** parameter to determine the relevance of the abstracts to the query. Results are sorted by relevance and returned according to the number specified by **choice_top_k**. The **format_node_batch_fn** and **parse_choice_select_answer_fn** functions can also be specified as parameters. The first function, **format_node_batch_fn**, prepares the information from nodes in a format suitable for the LLM. This may include combining text from multiple nodes, structuring the information in a particular way, or adding contextual elements to help the LLM understand and evaluate the content. The second function, **parse_choice_select_answer_fn**, can, for example, determine which nodes are most relevant to the query and extract relevance scores or other metrics associated with each node. By analyzing the LLM response, this function allows the retriever to decide which nodes are most relevant to the user's query. To summarize, **DocumentSummaryIndexLLMRetriever** is useful for retrieving useful data from a large number of documents using the natural language processing power of LLMs. As a useful side note, it is good to know that this retriever also returns the relevance score that is associated with each of the nodes.

**Additional observation**

During my experimentation with this type of retriever, I noticed that the relevance scores that are assigned to each node by the LLM were consistently high, often reaching the maximum value of 10 (tested using GPT3.5-Turbo). For applications where nuanced differentiation between degrees of relevance is crucial, it might be beneficial to adjust the prompt or apply post-processing to the LLM's responses to achieve a more balanced and nuanced distribution of relevance scores. This issue also underscores the importance of tailoring LLM prompts and response handling to suit the specific needs and contexts of different applications. We'll talk more about prompt customization in *Chapter 10*.

### DocumentSummaryIndexEmbeddingRetriever

To build this retriever, we can use the following code:

```python
DocumentSummaryIndex.as_retriever(
    retriever_mode='embedding'
)
```

This retriever relies on embeddings to retrieve summary nodes from the index. *Figure 6.6* exemplifies its operation:

![Figure 6.6 – DocumentSummaryIndexEmbeddingRetriever](images/chapter-06-figure-06-6-documentsummaryindexembeddingretriever.jpg)

*Figure 6.6 – DocumentSummaryIndexEmbeddingRetriever*

It computes the embeddings for the query and then finds the summaries with the highest similarity to the query. For this method to work, the index should have been built with the **embed_summaries** parameters set to **True**. The **similarity_top_k** parameter determines the number of summaries to return, based on their similarity to the query.

## Building more advanced retrieval mechanisms

Now that we've covered the basic retrievers, let's explore more advanced retrieval mechanisms that can help us build more sophisticated RAG applications.

### Increasing efficiency with asynchronous retrieval

For applications that need to handle multiple queries simultaneously or work with large datasets, asynchronous retrieval can significantly improve performance. LlamaIndex supports asynchronous operations for most retrievers.

```python
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

async def async_retrieval_example():
    documents = SimpleDirectoryReader("files").load_data()
    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever()

    # Perform multiple retrievals asynchronously
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?"
    ]

    tasks = [retriever.aretrieve(query) for query in queries]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results):
        print(f"Query {i+1}: {queries[i]}")
        print(f"Result: {result[0].text[:100]}...")
        print("---")

# Run the async function
asyncio.run(async_retrieval_example())
```

### Working with metadata filters

Metadata filters allow you to narrow down search results based on document properties. This is particularly useful for implementing access controls or domain-specific filtering.

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

# Create documents with metadata
documents = [
    Document(
        text="This is a technical document about machine learning.",
        metadata={"category": "technical", "department": "engineering", "level": "advanced"}
    ),
    Document(
        text="This is a business document about quarterly results.",
        metadata={"category": "business", "department": "finance", "level": "basic"}
    ),
    Document(
        text="This is a technical document about data structures.",
        metadata={"category": "technical", "department": "engineering", "level": "intermediate"}
    )
]

index = VectorStoreIndex.from_documents(documents)

# Create metadata filters
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="technical", operator=FilterOperator.EQ),
        MetadataFilter(key="level", value="advanced", operator=FilterOperator.EQ)
    ]
)

# Use filters in retrieval
retriever = index.as_retriever(filters=filters)
results = retriever.retrieve("machine learning algorithms")

for result in results:
    print(f"Text: {result.node.text}")
    print(f"Metadata: {result.node.metadata}")
    print("---")
```

### Using tools and selectors

LlamaIndex provides tools and selectors that can help route queries to the most appropriate retriever or index.

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import RetrieverTool
from llama_index.core.query_engine import RouterQueryEngine

# Create multiple indexes
documents = SimpleDirectoryReader("files").load_data()
vector_index = VectorStoreIndex.from_documents(documents)
summary_index = SummaryIndex.from_documents(documents)

# Create retriever tools
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_index.as_retriever(),
    description="Useful for semantic search and finding specific information"
)

summary_tool = RetrieverTool.from_defaults(
    retriever=summary_index.as_retriever(),
    description="Useful for getting comprehensive overviews and summaries"
)

# Create router query engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool]
)

# Use the router
response = query_engine.query("Give me a comprehensive overview of the documents")
print(response)
```

### Transforming queries and generating sub-queries

Sometimes, complex queries need to be broken down into simpler sub-queries for better results.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.question_gen.openai import OpenAIQuestionGenerator

# Create query engine tools for different domains
tech_index = VectorStoreIndex.from_documents(tech_documents)
business_index = VectorStoreIndex.from_documents(business_documents)

tech_tool = QueryEngineTool.from_defaults(
    query_engine=tech_index.as_query_engine(),
    description="Technical documentation and API references"
)

business_tool = QueryEngineTool.from_defaults(
    query_engine=business_index.as_query_engine(),
    description="Business processes and policies"
)

# Create sub-question query engine
question_gen = OpenAIQuestionGenerator.from_defaults()
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tech_tool, business_tool],
    question_gen=question_gen
)

# Use complex query that will be broken down
response = sub_question_engine.query(
    "How do our technical APIs integrate with our business processes, and what are the security implications?"
)
print(response)
```

## Understanding the concepts of dense and sparse retrieval

One of the most important concepts in modern information retrieval is the distinction between dense and sparse retrieval methods. Understanding when to use each approach can significantly impact the performance of your RAG system.

### Dense retrieval

Dense retrieval uses vector embeddings to represent both queries and documents in a high-dimensional space. This approach excels at capturing semantic meaning and can find relevant documents even when they don't share exact keywords with the query.

**Advantages of dense retrieval:**
- Excellent semantic understanding
- Can handle synonyms and paraphrasing
- Good for conceptual queries
- Robust to vocabulary mismatch

**Disadvantages of dense retrieval:**
- May miss exact keyword matches
- Requires pre-computed embeddings
- Can be computationally expensive
- May struggle with very specific technical terms

### Sparse retrieval

Sparse retrieval, exemplified by algorithms like BM25, relies on exact keyword matching and statistical measures of term importance. It creates sparse vectors where most dimensions are zero, representing only the terms that appear in the document.

**Advantages of sparse retrieval:**
- Excellent for exact keyword matching
- Fast and efficient
- Good for technical/domain-specific queries
- Interpretable results

**Disadvantages of sparse retrieval:**
- Limited semantic understanding
- Struggles with synonyms
- Sensitive to vocabulary mismatch
- May miss conceptually related content

### BM25Retriever example

Let's look at an example of how we can use **BM25Retriever**.

To use this particular retriever, you'll need to install the required Python package and the corresponding LlamaIndex integration package by running the following commands:

```bash
pip install rank-bm25
pip install llama-index-retrievers-bm25
```

After installing the **rank-bm25** package, you can test it with this sample code:

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader('files')
documents = reader.load_data()

splitter = SentenceSplitter.from_defaults(
    chunk_size=60,
    chunk_overlap=0,
    include_metadata=False
)

nodes = splitter.get_nodes_from_documents(documents)
```

We're using the two initial sample files containing data about ancient Rome and different breeds of dogs. In this example, I've used **SentenceSplitter**, configured with a relatively small chunk size. That is because the sample file is small in size and I wanted to produce more granular nodes structured as sentences to better exemplify the workings of **BM25Retriever**. Next, let's implement the retriever:

```python
retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2
)

response = retriever.retrieve("Who built the Colosseum?")

for node_with_score in response:
    print('Text:'+node_with_score.node.text)
    print('Score: '+str(node_with_score.score))
```

After chunking the two documents, we use the retriever to apply the BM25 algorithm and retrieve the two most relevant chunks relative to our query about the Colosseum.

You can further experiment with this sample and try to adjust the **similarity_top_k** parameter, the query, or the chunking strategy to better understand how this retriever works.

### When should we use sparse retrieval instead of dense retrieval?

Let's consider a practical example of when sparse retrieval might give better results than dense retrieval in an RAG application.

**A practical use case for sparse retrieval**

Suppose we've built a system for retrieving legal documents. In this scenario, user queries would likely include precise legal terms, citations, or specific phrases found in legal texts. Let's assume a user inputs a query such as, "*Article 45 of the GDPR regarding personal data transfers on the basis of an adequacy decision.*" This query contains specific phrases, such as "Article 45" and "GDPR," which are likely to be found in relevant legal documents exactly in this form.

Sparse search is likely to provide very accurate results for such a query. It will accurately locate documents that contain the specific article from the GDPR, reducing noise and irrelevant retrievals. Given that legal documents often have a structured format, with different sections and articles, sparse retrieval methods can efficiently parse through this structured data and retrieve nodes based on direct references found in the query.

Because dense retrieval methods tend to prioritize general meaning over exact term matching, they may produce less accurate results in such a specialized, keyword-specific query.

Unless trained specifically on legal texts, an embedding model used for dense retrieval might struggle to accurately interpret and match the complex legal jargon and specific citation styles used in legal queries.

### When would dense retrieval be a better choice?

Here's another practical example.

A typical use case where dense retrieval would most likely produce better results would be a customer support chatbot designed to understand and respond to a wide range of customer queries. Let's say the chatbot is tasked with assisting users with various issues related to technical products, such as hardware troubleshooting, software features, usage tips, and general inquiries about products and services.

A user might ask a question such as "*My laptop battery is draining really quickly, even when I'm not using it much. What can I do about it?*" Because dense search excels at understanding the semantic context of queries, in this case, it could understand the broader meaning behind phrases such as "battery drains really fast" and relate them to similar problems, even if the exact phrase isn't in the knowledge base.

Sparse methods, on the other hand, may not perform well if the query doesn't contain specific keywords that are present in the support documents. In our example, the user might describe a problem using different terms to those used in the technical manuals or FAQs.

### Can we combine the two methods in a single retriever?

The short answer is yes. You've probably already guessed that I'm building a case along these lines. By combining them, we'd get the best of both worlds in terms of benefits and features. A few pages ago, we talked about using selectors and routers to implement more complex query behavior in our RAG application.

I'll leave it up to you to adapt the methods I've demonstrated there and implement a hybrid system that uses both dense and sparse retrieval methods. If you feel the need for an additional example, you can have a look at this one, which uses the Pinecone vector database to implement hybrid search: https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo-Hybrid.html.

### Dealing with empty results from the retrieval process

Sometimes, our retrievers may come up empty-handed, without finding any indexed content matching the current query. This typically means that there are no relevant nodes in the index for that particular query.

In such cases, the retriever may return an empty result set, indicating that no matching nodes were found. Depending on the type of index used, this situation can arise if the query keywords are very specific or rare, and none of the nodes in the index contain those exact keywords, or, in the case of embedding-based indexes, the similarity search that was performed during the search did not find any matching nodes with the current parameters used. To handle this scenario, we can consider various approaches:

- **Fallback mechanisms**: The search system can have fallback strategies in place, such as performing a more general search by adjusting the retriever's parameters or suggesting alternative query terms to the user.

- **Query expansion**: The query can be automatically expanded to include synonyms, related terms, or broader concepts to increase the chances of finding relevant nodes.

- **Relevance scoring**: Even if no exact keyword matches are found, the search system can employ relevance scoring algorithms to identify nodes that are semantically similar to the query or contain partial matches.

## Discovering other advanced retrieval methods

In addition to the basic concepts just discussed, several other advanced retrieval methods are worth familiarizing yourself with. There is a special section in the official documentation where these methods are explained: https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/advanced_retrieval.html.

There, you will learn more about special techniques, such as *Small-to-Big retrieval*, *recursive retrieval*, *retrieval from embedded tables*, *multi-modal retrieval*, *auto-merging retrieval*, and others.

A detailed explanation of each retrieval strategy would go far beyond what I intend to cover in this book, but that doesn't mean they aren't important. After all, there is no point in ingesting and indexing the original documents if we cannot effectively extract the context we need in RAG.

**Practical advice**

Always read the latest version of the official documentation before starting a major project. Things move so fast, and new methods and techniques emerge so quickly, that it is a shame to waste time reinventing the wheel. As an anecdote, I can tell you from personal experience that I have spent hours *inventing* something very similar to the *small-to-big* method, only to discover a few days later that it was already a tested and documented technique.

That's enough information for one chapter. We'll skip the PITS coding practice now as we'll let more information accumulate in the next chapter before implementing additional features in our personal tutoring project.

## Summary

In this chapter, we explored various querying strategies and architectures within LlamaIndex with a deep focus on retrievers. Retrievers provide essential capabilities for extracting relevant information from indexes to generate useful responses in RAG systems. Throughout this chapter, we looked at basic retriever types such as **VectorIndexRetriever** and **SummaryIndexRetriever**. We also gained an understanding of advanced concepts such as asynchronous retrieval, metadata filters, tools, selectors, and query transformations. These allow us to build more sophisticated retrieval logic.

Additionally, we covered fundamental paradigms such as dense retrieval and sparse retrieval and discussed their strengths and weaknesses. Implementations in LlamaIndex such as BM25Retriever were also introduced.

Overall, this chapter provided an overview of retrieval capabilities in LlamaIndex, laying the foundation for building high-performance and contextually-aware RAG applications.

We're now equipped with the necessary knowledge to effectively retrieve information from indexes. In the next chapter, we'll build on this knowledge by addressing the other important components of a query engine: post-processors and response synthesizers.
