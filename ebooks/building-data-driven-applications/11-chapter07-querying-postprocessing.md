# Chapter 7: Querying Our Data, Part 2 – Postprocessing and Response Synthesis

Building on the knowledge acquired in the previous chapter, we will now explore various postprocessing techniques to refine the retrieved context before covering the final query response synthesis. Afterward, we will learn how to bring all these components together into powerful query engines so that we can perform end-to-end natural language querying over documents. We'll also get to practice our new skills by working on our tutoring project.

In this chapter, we're going to cover the following main topics:

- Re-ranking, transforming, and filtering nodes using postprocessors
- Understanding the response synthesizers
- Implementing output parsing techniques
- Building and using query engines
- Hands-on – building quizzes in PITS

## Technical requirements

For this chapter, you will need to install the following packages in your environment:

- **spaCy**: https://spacy.io/
- **Guardrails-AI**: https://www.guardrailsai.com/
- **pandas**: https://pandas.pydata.org/

All the code samples in this chapter can be found in the **ch7** subfolder of this book's GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

## Re-ranking, transforming, and filtering nodes using postprocessors

In the previous chapter, we discussed the various retrieval methods that LlamaIndex offers. We extracted the necessary context to be able to enrich and improve the query we are now sending to the LLM. But is this enough?

As we have already discussed, *naive* retrieval methods are unlikely to produce ideal results in any scenario. There will probably be many situations where the returned nodes will perhaps contain irrelevant information or will not be sorted in chronological order. These kinds of situations could put the LLM in difficulty, adversely affecting the quality of the prompt that our RAG application builds.

**A quick side notes**

In case it wasn't already obvious, the main purpose of a RAG flow is to programmatically build prompts. Instead of manually building these prompts and then inputting them into a ChatGPT-like interface, LlamaIndex dynamically assembles the prompts from our documents, which are split into nodes and then indexed and selected using retrievers. Many things could go wrong in this process. Maybe we didn't ingest the original documents completely or correctly, or maybe we didn't choose the right **chunk_size** value and ended up with nodes that were too granular or too loaded with irrelevant information. Maybe we didn't index them correctly, or maybe the retriever we used simply didn't select the nodes in the correct order or brought in more information than we wanted.

There are many points where errors could creep into the whole process. That doesn't sound very encouraging, does it?

The good news is that we still have an opportunity to improve this context before the final step of sending the information to the LLM. This opportunity comes in the form of **node postprocessors** and **response synthesizers**.

But first, let's understand how postprocessors work.

Node postprocessors are critical in refining the results that are obtained from the retrieval process. That is because no matter how good the retrieval step is, there is always a chance of additional, unnecessary retrieved data *polluting* our context and confusing the LLM. In other cases, the retrieved nodes might be relevant but not necessarily in the correct order, and that can also affect the quality of the LLM's response.

*Figure 7.1* depicts the role of the postprocessors in a RAG workflow:

![Figure 7.1 – The role of node postprocessors in RAG](images/chapter-07-figure-07-1-role-node-postprocessors-rag.jpg)

*Figure 7.1 – The role of node postprocessors in RAG*

These processors operate on a set of nodes, applying transformations or filters to enhance the relevance and quality of the information. They can be used on their own, to process a given set of nodes, but they are more commonly used within query engines, after the node retrieval step and before response synthesis. LlamaIndex provides various built-in processors but also the option of building custom postprocessing logic.

Let's begin by understanding the different purposes and operating modes of node postprocessors.

## Exploring how postprocessors filter, transform, and re-rank nodes

At their core, all node postprocessors work by adjusting the retrieved context before that context gets injected into a prompt and sent to the LLM for response synthesis. They operate by either filtering, transforming, or re-ranking nodes. Let's have a look at these operating modes to get a better understanding.

### Node filtering postprocessors

Node filtering postprocessors are designed to remove irrelevant or unnecessary nodes from the set of retrieved results. They work by applying specific criteria to each node and discarding those that don't meet the requirements. For example, **SimilarityPostprocessor** filters out nodes whose similarity score falls below a specified threshold, ensuring that only highly relevant nodes are passed to the language model for response generation. Similarly, **KeywordNodePostprocessor** keeps only the nodes that contain certain required keywords or excludes nodes with specific unwanted keywords. Node filtering helps to reduce information overload and improve the quality of the final response by focusing on the most pertinent information.

### Node transforming postprocessors

Node transforming postprocessors modify the content of the retrieved nodes without necessarily removing any of them. These postprocessors aim to enhance the relevance and usefulness of the information within each node. One example is **MetadataReplacementPostprocessor**, which replaces the content of a node with a specific field from that node's metadata. This allows the text being used to be dynamically adjusted to represent a node based on its metadata rather than the original ingested content. Another example is **SentenceEmbeddingOptimizer**, which optimizes longer text passages by selecting the most relevant sentences within a node based on their semantic similarity to the query. By transforming the nodes' content, these postprocessors help align the information more closely with the user's query and improve the overall quality of the generated response.

### Node re-ranking postprocessors

These postprocessors don't specifically remove or change the retrieved nodes. The purpose of a re-ranker is to take the initial set of nodes returned by the retriever and reorder them based on their relevance to the given query. This is particularly important when dealing with long-form queries or complex information needs as many LLMs struggle to effectively process and generate accurate responses when provided with lengthy or multi-faceted contexts. By employing a re-ranker, the RAG system can prioritize the most pertinent information and present it to the LLM in a more coherent format, thus leading to better responses.

Re-rankers often leverage advanced techniques such as deep learning, transformers, or LLMs themselves to assess the relevance of each retrieved document or passage. They may consider factors such as semantic similarity, context overlap, or query-document alignment to assign relevance scores to the retrieved nodes. The top-ranked nodes are then fed into the LLM, which generates the final response based on this refined context, enhancing the overall performance and utility of the RAG system. By incorporating a re-ranking step into the RAG pipeline, the system can overcome the limitations of LLMs in handling long or complex queries, ultimately providing more accurate, relevant, and useful responses to users.

Next, we'll explore the built-in LlamaIndex postprocessors in all three categories.

## SimilarityPostprocessor

**SimilarityPostprocessor** filters nodes by comparing them to a similarity score threshold. Nodes that score below this threshold are removed, ensuring only relevant and similar content to the query remains. This is particularly useful because it ensures that the nodes that are passed to the language model for response generation are relevant by having a high degree of semantic correlation with the query.

**A potential use cases**

An e-commerce company has a customer support chatbot powered by an LLM. Let's assume that the chatbot retrieves nodes from **KeywordTableIndex** and tries to identify all contexts based on the keywords contained in the user query. For a query such as, *How do I return a damaged item I received yesterday?*, the retrieved nodes might include general return policies, product descriptions for items ordered by the customer, shipping information, and even irrelevant product advertisements or promotions. **SimilarityPostprocessor** could filter out nodes that are not closely related to the specific context of the query. In this case, it would prioritize nodes specifically discussing return policies for damaged items and recent orders by the customer, while discarding general product advertisements and unrelated shipping details. That would greatly increase the chance of the LLM producing a more meaningful response.

This postprocessor takes a list of nodes, typically fetched by a retriever, as input, each with an associated similarity score. The postprocessor can be configured with a **similarity_cutoff** parameter. This threshold determines the minimum score a node must have to be considered relevant. If a node's similarity score is **None** or if it's lower than **similarity_cutoff**, the node is considered not to meet the threshold and is therefore excluded from the final list. Essentially, this postprocessor filters out any nodes that have a similarity score below the set threshold. This ensures that only nodes closely related to the query are retained. The nodes meeting or exceeding the similarity score threshold is then passed on for further processing or response synthesis. Here's a simple example of how we can use it in practice:

```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

reader = SimpleDirectoryReader('files/other')
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(retriever_mode='default')

nodes = retriever.retrieve(
    "What did Fluffy found in the gentle stream?"
)
```

In the first part of the code, we took care of the imports and then ingested a sample file into a document. Then, we created a **VectorStoreIndex** index and used the default retriever to fetch relevant nodes based on a query:

```python
print('Initial nodes:')
for node in nodes:
    print(f"Node: {node.node_id} – Score: {node.score}")
```

Here, we printed the original list of nodes since they were fetched by the retriever. Now, let's apply the postprocessor.

```python
pp = SimilarityPostprocessor(
    nodes=nodes,
    similarity_cutoff=0.86
)

remaining_nodes = pp.postprocess_nodes(nodes)

print('Remaining nodes:')
for node in remaining_nodes:
    print(f"Node: {node.node_id} – Score: {node.score}")
```

After building and applying the postprocessor on the nodes, we print the remaining nodes. The output will be similar to the following:

```
Initial nodes:
Node: da51464d-e83f-4aec-a9db-8bd839ab3a4c - Score: 0.8516122822966049
Node: f839ec27-e487-4132-b139-79e3695d5500 - Score: 0.8368901228748273

Remaining nodes:
Node: da51464d-e83f-4aec-a9db-8bd839ab3a4c - Score: 0.8516122822966049
```

As we can see, the second node from the initial list was removed because it had a score below the threshold we defined – 0.85.

## KeywordNodePostprocessor

**KeywordNodePostprocessor** is designed to refine the selection of nodes based on specific keywords. This postprocessor works by ensuring that the retrieved nodes either contain certain required keywords or exclude specific unwanted keywords. It's a great method for aligning the content of the nodes more closely with the user's query by focusing on keyword relevance.

**Practical use case in a RAG scenario**

Imagine a scenario in a corporate environment where the RAG system is used to retrieve information from a vast internal database for employee queries. However, there are certain confidential files or sections of files that should not be accessible to all employees. By configuring **KeywordNodePostprocessor** with keywords that indicate sensitive content (such as *confidential*, *restricted*, or specific project code names), the system can automatically exclude nodes containing these keywords from the retrieval results. This setup ensures that sensitive information is not inadvertently disclosed, maintaining the integrity and confidentiality of the corporate data.

It takes a list of nodes as input, typically fetched by a retriever, and is configured with parameters for required and excluded keywords. **KeywordNodePostprocessor** then processes these nodes, keeping only those that meet the keyword criteria. This ensures that the final set of nodes is highly relevant to the specific query, leading to more accurate and useful responses in a RAG system.

**Quick note**

The postprocessor relies on the **spaCy** library (https://pypi.org/project/spacy/), which you must install on your system before running the next example. This is a powerful Python library for advanced NLP. Its features include neural network models for various NLP tasks such as tagging, parsing, and NER. It's a piece of commercial open source software available under an MIT license.

To use **KeywordNodePostprocessor**, make sure you install spaCy in your environment by running the following command:

```bash
pip install spacy
```

Here's a basic example of how to use this postprocessor to filter out some log entries based on their classification label:

```python
from llama_index.core.postprocessor import KeywordNodePostprocessor
from llama_index.core.schema import TextNode, NodeWithScore

nodes = [
    TextNode(
        text="Entry no: 1, <SECRET>, Attack at Dawn"
    ),
    TextNode(
        text="Entry no: 2, <RESTRICTED>, Go to point Bravo"
    ),
    TextNode(
        text="Entry no: 3, <PUBLIC>, text: Roses are Red"
    ),
]
```

In this example, we're manually defining the nodes instead of ingesting data from external files. After we define the nodes, we have to wrap them into **NodeWithScore** because that's the expected input of the postprocessor:

```python
node_with_score_list = [
    NodeWithScore(node=node) for node in nodes
]

pp = KeywordNodePostprocessor(
    exclude_keywords=["SECRET", "RESTRICTED"]
)

remaining_nodes = pp.postprocess_nodes(
    node_with_score_list
)

print('Remaining nodes:')
for node_with_score in remaining_nodes:
    node = node_with_score.node
    print(f"Text: {node.text}")
```

In this example, **KeywordNodePostprocessor** filters the nodes fetched by the retriever, excluding those that include **SECRET** and **RESTRICTED**.

Several parameters can be customized with this postprocessor. The most important ones are as follows:

- **required_keywords**: This is a list of strings, where each string represents a keyword that must be present in the node for it to be included in the final output. If this list is not empty, the postprocessor will filter out any nodes that do not contain these keywords.

- **exclude_keywords**: Similar to **required_keywords**, this is also a list of strings. However, in this case, any node containing a keyword from this list will be excluded from the final output. It's used for filtering out nodes based on unwanted content.

- **lang**: This argument specifies the language model to be used by the internal spaCy NLP library. The default value is *en* for English, but it can be set to other language codes supported by Spacy. The effectiveness and accuracy of keyword matching might depend on the language-specific processing of the text. For example, the way words are tokenized by Spacy can affect how keywords are identified.

Keep in mind that keywords – both required and excluded – are processed in a case-sensitive way. To ensure consistent behavior regardless of case, you might consider converting both the keywords and the text in the nodes into the same case (for example, all lowercase) before processing.

## PrevNextNodePostprocessor

**PrevNextNodePostprocessor** is designed to enhance node retrieval by fetching additional nodes based on their relational context in the document. This postprocessor can operate in three modes – **previous**, **next**, or **both** – allowing users to retrieve nodes that are either preceding, succeeding, or both concerning the current set of nodes.

**A potential use cases**

Consider a legal research scenario where a user queries a RAG system about a specific legal case. **PrevNextNodePostprocessor** can be set in *both* modes to retrieve not only the nodes directly related to the case but also the preceding and succeeding nodes that might contain vital contextual information, such as related legal precedents or subsequent rulings. This ensures a comprehensive understanding of the case by providing a broader context, which is especially crucial in legal research where every detail matters.

The process begins by taking a list of nodes, typically fetched by a retriever. It then extends this list by adding nodes that are directly preceding, succeeding, or both, based on the configured mode. This results in a more contextually enriched set of nodes, leading to responses that are more nuanced and comprehensive in a RAG system. Here's a list of the parameters for this postprocessor:

- **docstore**: The actual document store storing the nodes.
- **num_nodes**: This sets the number of nodes to return. By default, it returns 1 node in the chosen direction.
- **mode**: Can be set to previous, next, or both.

Additionally, we have **AutoPrevNextNodePostprocessor**, which is an advanced variation of **PrevNextNodePostprocessor**. This one is intelligently inferring whether to fetch additional nodes based on the *previous*, *next*, or neither relationship in response to the query context.

In comparison to **PrevNextNodePostprocessor**, which requires manual setting for mode selection, **AutoPrevNextNodePostprocessor** automates this process. It utilizes specific prompts to infer the direction (previous, next, or none) based on the current context and the query.

This inference is particularly useful in scenarios where the direction of node retrieval isn't explicitly clear or when it needs to be dynamically determined based on the nature of the query and existing answers. For example, in a scenario where a RAG system is used for historical research, **AutoPrevNextNodePostprocessor** can automatically determine whether to fetch preceding or succeeding historical events or data points based on the query's context, enhancing the relevance and comprehensiveness of the response.

This capability makes it useful in applications where the sequence of information and its contextual relevance are essential for generating accurate and useful responses.

The prompts can be customized using the **infer_prev_next_tmpl** and **refine_prev_next_tmpl** arguments. There's also a **Verbose** argument, which provides more visibility on the selection process.

## LongContextReorder

**LongContextReorder** is specifically designed to improve the performance of LLMs in handling long context scenarios. Research has shown that significant details in extended contexts are better utilized when positioned at the start or end of the input context *(Liu et al., Lost in the Middle: How Language Models Use Long Contexts (2023)* – https://arxiv.org/abs/2307.03172). The **LongContextReorder** postprocessor addresses this by reordering the nodes, placing crucial information where it's more accessible to the model.

**A practical scenario**

In a RAG system, particularly in academic or research-oriented queries where long, detailed documents are common, **LongContextReorder** can be very useful. For instance, if a user queries about detailed historical events, the system might retrieve lengthy nodes encompassing extensive details. **LongContextReorder** would rearrange these nodes, ensuring that the most relevant details are positioned at the beginning or end, thereby enhancing the model's ability to extract and utilize this crucial information effectively. This results in responses that are more coherent and contextually rich, significantly improving the overall quality of the output in cases involving lengthy contexts.

**LongContextReorder** takes a list of nodes, typically fetched by a retriever, and reorders them based on their relevance scores. The goal is to optimize the arrangement of information in a way that maximizes the language model's ability to access and process significant details, especially in cases where the context length might otherwise hinder performance.

This postprocessor is particularly effective in scenarios where detailed and comprehensive responses are required, ensuring that the most relevant information is presented in a way that is most accessible to the model.

## PIINodePostprocessor and NERPIINodePostprocessor

These postprocessors mask **personally identifiable information** (**PII**) in nodes, improving privacy and security. **PIINodePostprocessor** is designed to use a local model, while **NERPIINodePostprocessor** relies on a NER model from Hugging Face. We saw an example of how this postprocessor works in *Chapter 4*, *Ingesting Data into Our RAG Workflow*, in the *Scrubbing personal data and other sensitive information* section.

**PIINodePostprocessor** takes the following arguments:

- **llm**: This object should contain a local model for processing.
- **pii_str_tmpl**: This can be used to customize the default prompt template used for masking personal data.
- **pii_node_info_key**: This string serves as a key in the node's metadata to store information related to PII processing. It's used to track and reference the PII data processed within each node. It can be used to later recompose the original information if required.

**NERPIINodePostprocessor** can be configured with the **pii_node_info_key** parameter. Similar to the previous postprocessor, this string key is used to store information related to PII processing in the node's metadata. It's a unique identifier within the node metadata for tracking the PII data that has been processed.

## Understanding response synthesizers

After retrieval and postprocessing, the final step in the RAG pipeline is response synthesis. This is where the LLM takes the refined context and generates a coherent, relevant response to the user's query.

Response synthesizers are responsible for taking the retrieved and processed nodes and combining them with the original query to generate a final response. LlamaIndex provides several different response synthesis strategies, each optimized for different scenarios and requirements.

### Types of response synthesizers

#### TreeSummarize

**TreeSummarize** builds a tree of summaries over the set of retrieved nodes. It works by recursively summarizing pairs of nodes until a single summary remains. This approach is particularly effective for handling large amounts of context that might exceed the LLM's context window.

```python
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)

synthesizer = TreeSummarize()
query_engine = index.as_query_engine(response_synthesizer=synthesizer)

response = query_engine.query("What are the main topics covered?")
print(response)
```

#### Refine

The **Refine** synthesizer iteratively refines an answer by going through each retrieved node sequentially. It starts with an initial answer based on the first node, then refines this answer by incorporating information from subsequent nodes.

```python
from llama_index.core.response_synthesizers import Refine

synthesizer = Refine()
query_engine = index.as_query_engine(response_synthesizer=synthesizer)

response = query_engine.query("Explain the key concepts in detail")
print(response)
```

#### CompactAndRefine

This synthesizer combines the benefits of both compacting and refining. It first compacts the retrieved nodes to fit within the context window, then applies the refine strategy to generate the final response.

```python
from llama_index.core.response_synthesizers import CompactAndRefine

synthesizer = CompactAndRefine()
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
```

#### SimpleSummarize

**SimpleSummarize** concatenates all retrieved nodes and sends them to the LLM in a single prompt. This is the fastest approach but may not work well if the total context exceeds the LLM's context window.

```python
from llama_index.core.response_synthesizers import SimpleSummarize

synthesizer = SimpleSummarize()
query_engine = index.as_query_engine(response_synthesizer=synthesizer)
```

### Customizing response synthesis

You can customize various aspects of response synthesis:

```python
from llama_index.core.response_synthesizers import get_response_synthesizer

synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    summary_template="Summarize the following text: {context_str}",
    text_qa_template="Context: {context_str}\nQuestion: {query_str}\nAnswer:",
    streaming=True
)
```

## Implementing output parsing techniques

Output parsing ensures that the LLM's responses are structured and formatted according to specific requirements. This is particularly important when building applications that need to process the LLM's output programmatically.

### Pydantic output parsing

LlamaIndex integrates with Pydantic to provide structured output parsing:

```python
from pydantic import BaseModel, Field
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

class BookSummary(BaseModel):
    title: str = Field(description="The title of the book")
    author: str = Field(description="The author of the book")
    main_themes: list[str] = Field(description="List of main themes")
    rating: int = Field(description="Rating from 1-10")

# Create output parser
output_parser = PydanticOutputParser(BookSummary)

# Create query engine with output parser
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(output_parser=output_parser)

response = query_engine.query("Summarize this book")
print(type(response.response))  # BookSummary object
print(response.response.title)
print(response.response.main_themes)
```

### Guardrails output parsing

For more advanced validation and correction, you can use Guardrails:

```bash
pip install guardrails-ai
```

```python
from llama_index.core.output_parsers import GuardrailsOutputParser
import guardrails as gd

# Define Guardrails spec
rail_spec = """
<rail version="0.1">
<output>
    <string name="summary" description="A brief summary" validators="length: 100 500" />
    <integer name="confidence" description="Confidence score" validators="range: 1 10" />
</output>
</rail>
"""

# Create output parser
output_parser = GuardrailsOutputParser.from_rail_string(rail_spec)

# Use with query engine
query_engine = index.as_query_engine(output_parser=output_parser)
response = query_engine.query("Provide a summary with confidence score")
```

### Custom output parsing

You can also create custom output parsers:

```python
from llama_index.core.output_parsers import BaseOutputParser

class CustomOutputParser(BaseOutputParser):
    def parse(self, output: str) -> dict:
        # Custom parsing logic
        lines = output.strip().split('\n')
        result = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        return result

    def format(self, query: str) -> str:
        return f"Please format your response as key:value pairs.\nQuery: {query}"

# Use custom parser
custom_parser = CustomOutputParser()
query_engine = index.as_query_engine(output_parser=custom_parser)

## Building and using query engines

Query engines are the high-level interface that combines all the components we've discussed: retrievers, postprocessors, and response synthesizers. They provide a simple API for end-to-end querying while allowing for extensive customization.

### Basic query engine usage

The simplest way to create a query engine is directly from an index:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create basic query engine
query_engine = index.as_query_engine()

# Query the engine
response = query_engine.query("What is the main topic of these documents?")
print(response)
```

### Customizing query engines

You can customize various components of the query engine:

```python
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer

# Create custom components
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# Create customized query engine
query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor],
    response_synthesizer=synthesizer,
    similarity_top_k=5
)
```

### RouterQueryEngine

For handling multiple data sources or different types of queries, you can use **RouterQueryEngine**:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# Create different indexes for different domains
tech_docs = SimpleDirectoryReader("tech_docs").load_data()
business_docs = SimpleDirectoryReader("business_docs").load_data()

tech_index = VectorStoreIndex.from_documents(tech_docs)
business_index = VectorStoreIndex.from_documents(business_docs)

# Create query engine tools
tech_tool = QueryEngineTool.from_defaults(
    query_engine=tech_index.as_query_engine(),
    description="Technical documentation and API references"
)

business_tool = QueryEngineTool.from_defaults(
    query_engine=business_index.as_query_engine(),
    description="Business processes and policies"
)

# Create router query engine
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[tech_tool, business_tool]
)

# Use the router
response = router_query_engine.query("How do I configure the API settings?")
```

### SubQuestionQueryEngine

For complex queries that can be broken down into sub-questions:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.question_gen.openai import OpenAIQuestionGenerator

# Create sub-question query engine
question_gen = OpenAIQuestionGenerator.from_defaults()
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tech_tool, business_tool],
    question_gen=question_gen
)

# Use with complex query
response = sub_question_engine.query(
    "How do our technical APIs integrate with business processes and what are the security considerations?"
)
```

### RetrieverQueryEngine

For more control over the retrieval process:

```python
from llama_index.core.query_engine import RetrieverQueryEngine

# Create custom retriever
retriever = index.as_retriever(similarity_top_k=10)

# Create query engine with custom retriever
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[postprocessor],
    response_synthesizer=synthesizer
)
```

## Hands-on – building quizzes in PITS

One of the features we are building in our PITS project is the ability to generate quizzes based on the learning material uploaded by the user.

These quizzes will initially be used to gauge the overall knowledge of the user on the topic. Based on that assessment, the training slides and narration will be adjusted to the level of the learner.

The same mechanism can also be used to generate intermediate quizzes at the end of each section to test the user's current knowledge. Let's see how we can easily implement the quiz builder feature.

We'll be using one of the LlamaIndex pre-packaged pydantic programs: the DataFrame Pydantic extractor. This is designed to extract tabular DataFrames from raw text.

Let's have a look at the code in **quiz_builder.py**:

```python
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.program.evaporate.df import DFRowsProgram
from llama_index.program.openai import OpenAIPydanticProgram
from global_settings import INDEX_STORAGE, QUIZ_SIZE, QUIZ_FILE
import pandas as pd
```

First, we imported all the necessary modules, including our global variables defined in **global_settings.py**:

- **INDEX_STORAGE**: The index's storage location
- **QUIZ_SIZE**: The number of questions to be included in a quiz
- **QUIZ_FILE**: The path where the quiz will be saved as a CSV

We're also importing the **load_index_from_storage** function, which we will use to fetch our indexes from storage to avoid the cost and time of rebuilding them.

Because we're using DataFrames, we'll also need to import the pandas library. If you don't have it already installed in your environment, make sure you run this first:

```bash
pip install pandas
```

OK – let's build our main function. The **build_quiz** function will be responsible for generating the quiz and saving the questions in a **CSV** file for further use:

```python
def build_quiz(topic):
    df = pd.DataFrame({
        "Question_no": pd.Series(dtype="int"),
        "Question_text": pd.Series(dtype="str"),
        "Option1": pd.Series(dtype="str"),
        "Option2": pd.Series(dtype="str"),
        "Option3": pd.Series(dtype="str"),
        "Option4": pd.Series(dtype="str"),
        "Correct_answer": pd.Series(dtype="str"),
        "Rationale": pd.Series(dtype="str"),
    })
```

First, we set up a DataFrame to structure the quiz questions and their associated options and answers. This DataFrame will serve as the foundation for our quiz. It includes columns for the question number, question text, four answer options, the correct answer, and a rationale for the answer. The use of a pandas DataFrame will make handling and manipulating the quiz data much easier.

Next, we need to load our vector index from storage. To do this, we must define a **StorageContext** object while using the **INDEX_STORAGE** folder as a parameter:

```python
storage_context = StorageContext.from_defaults(
    persist_dir=INDEX_STORAGE
)
vector_index = load_index_from_storage(
    storage_context,
    index_id="vector"
)
```

Here, we used **index_id** to identify the *vector* index because there's also a **TreeIndex** index in that storage that we won't be using for now. It's time to initialize our **DataFrame** extractor:

```python
df_rows_program = DFRowsProgram.from_defaults(
    pydantic_program_cls=OpenAIPydanticProgram,
    df=df
)
```

Now, we can define our query engine and craft a prompt that will generate the quiz questions:

```python
query_engine = vector_index.as_query_engine()
quiz_query = (
    f"Create {QUIZ_SIZE} different quiz "
    "questions relevant for testing "
    "a candidate's knowledge about "
    f"{topic}. Each question will have 4 "
    "answer options. Questions must be "
    "general topic-related, not specific "
    "to the provided text. For each "
    "question, provide also the correct "
    "answer and the answer rationale. "
    "The rationale must not make any "
    "reference to the provided context, "
    "any exams or the topic name. Only "
    "one answer option should be correct."
)

response = query_engine.query(quiz_query)
```

Next, the prompt is passed to the query engine, and the response is then processed by **DFRowsProgram** to convert it into a structured DataFrame format:

```python
result_obj = df_rows_program(input_str=response)
new_df = result_obj.to_df(existing_df=df)
new_df.to_csv(QUIZ_FILE, index=False)
return new_df
```

Finally, the new DataFrame containing the quiz questions is saved as a CSV file in the path defined by **QUIZ_FILE**. The function returns the new DataFrame for further use.

This serves as a simple demonstration of how to leverage a combination of LlamaIndex features, Pydantic programs, and DataFrame manipulation to create a dynamic quiz generator. We'll continue working on the rest of the features in future chapters.

## Summary

This chapter explored how to refine search results with various postprocessors, generate responses using different synthesizers, and ensure structured outputs with specific parsers.

We also explored how to construct query engines while integrating the various components that we discussed in the previous chapters.

This chapter also covered handling diverse data sources with **RouterQueryEngine** and decomposing complex queries with **SubQuestionQueryEngine**, and also demonstrated quiz creation in our tutoring app.

See you in the next chapter, where we'll talk about chatbots, agents, and conversation tracking with LlamaIndex.
```
