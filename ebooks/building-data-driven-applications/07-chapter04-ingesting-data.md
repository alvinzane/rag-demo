# Chapter 4: Ingesting Data into Our RAG Workflow

We've taken a good look at the overall structure of LlamaIndex from afar. It's now time to get much closer and understand the small details of this framework. It's bound to get more technical but also more intriguing as we go further.

Ready to go deeper down the rabbit hole? Follow me!

In this chapter, we will learn about the following:

- Using the LlamaHub connectors to ingest our data
- Taking advantage of the many text-chunking tools in LlamaIndex
- Infusing our nodes with metadata and relationships
- Keeping our data private and our budget safe
- Creating ingestion pipelines for better efficiency and lower costs

## Technical requirements

You will need to install the following Python libraries in your environment to be able to run the examples included in this chapter:

- **LangChain**: https://www.langchain.com/
- **Py-Tree-Sitter**: https://pypi.org/project/tree-sitter/

In addition, several LlamaIndex Integration packages will be required:

- **Entity extractor**: https://pypi.org/project/llama-index-extractors-entity/
- **Hugging Face LLMs**: https://pypi.org/project/llama-index-llms-huggingface/
- **Database reader**: https://pypi.org/project/llama-index-readers-database/
- **Web reader**: https://pypi.org/project/llama-index-readers-web/

All the code examples in this chapter can be found in the *ch4* subfolder of this book's GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

## Ingesting data via LlamaHub

As we saw in *Chapter 3*, *Kickstarting Your Journey with LlamaIndex*, one of the first steps in a RAG workflow is to ingest and process our proprietary data. We already discovered the concepts of documents and nodes, which are used to organize the data and prepare it for indexing. I've also briefly introduced the LlamaHub data loaders as a way to easily ingest data into LlamaIndex. It's time to examine these steps in more detail and gradually learn how to infuse LLM applications with our own, proprietary knowledge. Before we continue, though, I'd like to emphasize some very common challenges encountered at this step:

- No matter how effective our RAG pipeline is, at the end of the day, the quality of the final result will largely depend on the quality of the initial data. To overcome this challenge, make sure you start by cleaning up your data first. Eliminate potential duplicates and errors. While not exactly duplicates, redundant information can also clutter your knowledge base and confuse the RAG system. Be on the lookout for ambiguous, biased, incomplete, or outdated information. I've seen many cases of poorly structured and insufficiently maintained knowledge repositories that were completely useless for users looking for quick and accurate answers. Ask yourself this question: *If I were to manually search through this data, how easy would it be to find the information I need?* Before moving on with building the pipeline, do yourself a favor and prepare your data thoroughly until you're satisfied with the answer to that question.

- Our data is dynamic. An organizational knowledge repository is rarely a static, permanent data source. It evolves with the business, reflecting new insights, discoveries, and changes in the external environment. Recognizing this fluid nature is key to maintaining a relevant and effective system. To overcome this challenge, in a production RAG application, you'll have to implement a systematic method for periodically reviewing and updating the content, ensuring that new information is incorporated and outdated or incorrect data is removed.

- Data comes in many flavors, shapes, and sizes. Sometimes, it's structured, sometimes not. A well-built RAG system should be able to properly ingest all kinds of formats and document types. While LlamaIndex provides a huge number of data loaders for many different APIs, databases, and document types, building an automated ingestion system can still prove to be challenging. To overcome this particular challenge, later in this section, we'll cover **LlamaParse** – an innovative hosted service designed to automatically ingest and process data from different data sources.

Now that we know what kind of problems await along the way, let's start our journey by first discussing the simplest ways of ingesting the data into the RAG pipeline – by using the available LlamaHub data loaders.

## An overview of LlamaHub

LlamaHub is an extensive library of integrations that augments the capabilities of the core framework. Among many other types of integrations, LlamaHub contains numerous **connectors** – also known as **data readers** or **data loaders** – specially built to allow seamless integration of external data with LlamaIndex. There are over 180 readily available data readers spanning a wide range of data sources and formats, and the list is constantly increasing.

These connectors act as a standard way to ingest data, extracting data from sources such as databases, APIs, files, and websites and converting it into LlamaIndex **Document** objects. This relieves you from the burden of writing customized parsers and connectors for every new data source. But of course, if you're not satisfied with the existing connectors, you can always build your own and contribute to the collection.

LlamaHub empowers you to tap into diverse data sources with just a few lines of code. The resulting Document objects can then be parsed into nodes and indexed as required by your application. The unified output as LlamaIndex **Document** objects means your core business logic does not have to worry about handling various data types. The complexity is abstracted by the framework.

Why do we need so many integrations?

In *Chapter 2*, *LlamaIndex: The Hidden Jewel - An Introduction to the LlamaIndex Ecosystem*, in the *Familiarizing ourselves with the structure of the LlamaIndex code repository* section, I explained the motives behind the framework's modular architecture. Because of this modular architecture, many RAG components provided by LlamaIndex are not included in the core elements that are installed together with the rest of the framework. This means that before using any data loader for the first time, we have to install the corresponding integration package. Once the package has been installed, we'll be able to import the reader into our code and use its functionality. Some readers also utilize specialized libraries and tools tailored to each data type. For example, **PDFReader** leverages Camelot and Tika for parsing PDF content. **AirbyteSalesforceReader** uses the Salesforce API client, and so on. This allows us to efficiently adapt to the format and interface of each source but may require us to install additional packages in our development environment.

All available readers are listed on the LlamaHub website and usually come with detailed documentation and usage samples. Therefore, I'll briefly cover just a few examples to give a general idea of how you can use them in your applications.

I strongly encourage you to take your time and go through the entire list of data readers when building your LlamaIndex apps instead of spending valuable time building one from scratch. Chances are you'll just be reinventing the wheel.

If you're looking to consult the source code for the readers, you'll find them all included in the Llama-index GitHub repository, under the **llama-index-integrations/readers** subfolder: https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/readers.

The LlamaHub documentation for each data reader lists its installation requirements and usage guidance, so before trying to use them, make sure you also install any additional dependencies required by specific connectors you want to use.

## Using the LlamaHub data loaders to ingest content

Apart from the *Wikipedia* reader that we discussed in the previous chapter, to get a better understanding of how data readers work, let's look at a few more examples of LlamaHub readers that we can use to ingest data.

### Ingesting data from a web page

**SimpleWebPageReader** can extract text content from web pages.

To use it, we must first install the corresponding integration:

```bash
pip install llama-index-readers-web
```

Once installed, it's really easy to use:

```python
from llama_index.readers.web import SimpleWebPageReader
urls = ["https://docs.llamaindex.ai"]
documents = SimpleWebPageReader().load_data(urls)
for doc in documents:
    print(doc.text)
```

This loads and displays the text content of the specified web pages into documents.

At its core, **SimpleWebPageReader** serves as a bridge between the vast, unstructured world of the internet and the structured environment of the LlamaIndex RAG pipeline. To better understand its inner workings, let's explore what happens under the hood when it extracts text content from web pages.

When loading the data, **SimpleWebPageReader** iterates over a list of URLs provided by the user. For each URL, it performs a web request to fetch the page content. The response, initially in HTML format, can be transformed into plain text if the **html_to_text** flag is set to **True**. This transformation strips away the HTML tags and converts the web page content into a more digestible text format. However, remember what I've said about external dependencies for these readers? In this case, the HTML-to-text conversion feature requires the **html2text** package, which has to be installed first.

Another significant aspect of this reader is its ability to attach metadata to the scraped documents. Through the **metadata_fn** parameter, we can pass a custom function that takes a URL as input and returns a dictionary of metadata. This flexibility allows for the enrichment of documents with additional information or any relevant tags that might be useful in categorizing and understanding the context of the data better. Should the user provide a **metadata_fn** parameter, the reader then applies this function to the current URL to extract metadata, enriching the final **Document** object with this additional layer of information.

A practical use case for the metadata_fn function

We could, for example, use a function that simply returns the current date and time. That way, we could ingest the same URL at different moments and build a chronological timeline highlighting different versions of that page at various points in time. This could prove useful in scenarios such as browsing a code repository or answering questions about a developing news story.

Finally, each web page's content, along with its URL and optionally added metadata, is encapsulated in a **Document** object. These objects are then collected into a list, providing a structured representation of the text content and metadata extracted from each web page.

One thing to keep in mind

As its name suggests, this reader is a simple tool. While it can be effective for reading simple web pages, for more advanced cases such as pages requiring interaction (for example, navigating a login process or handling JavaScript-rendered content), **SimpleWebPageReader** might not be sufficient. Websites that dynamically generate content based on user interactions or rely heavily on client-side scripting can pose challenges that this basic scraper is not designed to handle.

Through **SimpleWebPageReader**, the task of ingesting and structuring basic web content is simplified. The great thing about these readers is that they allow us to focus on building and enhancing the logic of our RAG applications instead of spending precious time on building compatible ingestion tools for each type of data in our knowledge base.

### Ingesting data from a database

Using databases is not only a common practice but also a highly efficient method for managing and retrieving structured information. Databases offer a robust platform for storing a vast array of data types, from simple text to complex relationships between entities, making them an indispensable asset in data management.

The **DatabaseReader** connector allows querying many database systems. First, we need to install the necessary integration package:

```bash
pip install llama-index-readers-database
```

Here's an example of how you can easily fetch the contents of an SQLite database:

```python
from llama_index.readers.database import DatabaseReader
reader = DatabaseReader(
    uri="sqlite:///files/db/example.db"
)
query = "SELECT * FROM products"
documents = reader.load_data(query=query)
for doc in documents:
    print(doc.text)
```

Under the hood, **DatabaseReader** connects to various databases to fetch data and transform it into a format usable by the RAG pipeline. It supports connection through a **SQLDatabase** instance, a **SQLAlchemy Engine**, a connection URI, or a set of database credentials – provided through the **scheme**, **host**, **port**, **user**, **password**, and **dbname** arguments. Once set up, it executes a provided SQL query to retrieve data. After connecting to the database, the reader executes the provided **query**. The resulting rows are then converted into Document objects, with each row from the query result forming a single Document. The conversion process involves concatenating each column-value pair into a string, which is then assigned as the text of a document.

The example I have provided executes the SQL query against an SQLite database stored in the **ch4/files/db** folder, loads each returned row as a Document, and displays the results. You can find a more general example on the official project documentation website: https://docs.llamaindex.ai/en/stable/examples/data_connectors/DatabaseReaderDemo.html.

Alright – I think you understand the workflow now. As you've probably noticed, the approach for using LlamaHub readers is very straightforward. In all the examples, first, we install the required integration package, as described on LlamaHub, and then use it to import and load data from the reader. Apart from the examples I have provided, you'll find a huge number of data readers available on LlamaHub. From Office documents, Gmail accounts, videos and images, YouTube videos, and RSS feeds to GitHub repositories and Discord chats, pretty much every popular data format is supported.

But apart from reading individual files using dedicated data readers, in the next section, we will also explore more efficient methods that can be used for ingesting multiple documents at once.

### Bulk-ingesting data from sources with multiple file formats

Loading data into LlamaIndex is a crucial first step. But sifting through the wide range of data loaders in LlamaHub and figuring out how to configure each one can feel overwhelming early on. That's why I'm going to show you two different methods that can greatly simplify and reduce the burden of data ingestion for your RAG systems.

We'll start with the simple method first.

#### Using SimpleDirectoryReader to ingest multiple data formats

When you just want to get started fast or have a simple use case, **SimpleDirectoryReader** comes to the rescue. Think of this reader as your trusty pocketknife for bulk data ingestion. It's easy to use, requires minimal setup, and automatically adapts to different file types. To load data, you simply point the reader to a folder or list of files. Loading a folder containing PDFs, Word docs, plain text files, and CSVs is very straightforward. Here's a demonstration:

```python
from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader(
    input_dir="files",
    recursive=True
)
documents = reader.load_data()
for doc in documents:
    print(doc.metadata)
```

Under the hood

**SimpleDirectoryReader** has built-in methods to determine which reader works best for each file type. You don't need to worry about those details. It will automatically detect formats such as PDF, DOCX, CSV, plain text, and others based on the file extensions. Then, it chooses the best tool to extract the content into Document objects. For plain text files, it simply reads the text content. For binary files such as PDFs and Office docs, it uses libraries such as PyPDF and Pillow to extract the text.

**SimpleDirectoryReader** effortlessly handles the different files and returns the parsed content as documents. By default, it only processes files in the directory's top level. To include subdirectories, you can set the **recursive** parameter to **True**.

You can also pass in a list of specific files to load, like this:

```python
files = ["file1.pdf", "file2.docx", "file3.txt"]
reader = SimpleDirectoryReader(files)
documents = reader.load_data()
```

The result is a batch of Document objects ready for indexing in just a few lines of code. No headaches setting up separate data readers for each file type. When you want quick and easy data ingestion without the complexity, let **SimpleDirectoryReader** handle the hard work! It's versatile and automated.

#### Parsing like a pro with the help of LlamaParse

While **SimpleDirectoryReader** is great for quick and easy data ingestion, sometimes, you need more advanced parsing capabilities, especially for complex file formats. Most of the time, we have to deal with complex file structures containing a mix of data. For example, a PDF file may include images, charts, code snippets, mathematical formulas, and other elements alongside its text content. The naive readers included in the LlamaHub integration library will be overwhelmed by such cases. They would most probably fail to extract the entire content or – even worse – mess up the extracted data and complicate its further processing.

This is where LlamaParse shines. Provided through the LlamaCloud enterprise platform (https://cloud.llamaindex.ai/parse), this reader is implemented through a cutting-edge hosted service that integrates seamlessly with the other components of the framework. It uses multi-modal capabilities and LLM intelligence under the hood to provide industry-leading document parsing, including exceptional support for tricky formats such as PDFs containing tables, figures, and equations.

One of the standout features of **LlamaParse** is that it allows you to provide natural language instructions to guide the parsing by using the **parsing_instruction** parameter. Since you know your documents best, you can tell **LlamaParse** exactly what kind of output you need and how that information should be extracted from the files.

For instance:

When parsing a technical whitepaper, you could instruct it to extract all the section headings, ignore the footnotes, and output any code snippets in markdown format. **LlamaParse** will follow your instructions to parse the document accurately.

In addition to the instruction-guided parsing mode, **LlamaParse** also offers a JSON output mode that provides rich structured data about the parsed document, including marking tables, headings, extracting images, and more. Also, for bulk-ingesting entire folders in one go, **LlamaParse** can be used in combination with **SimpleDirectoryReader**, as you will see in the next example. This gives you full flexibility to build custom RAG applications over a complex collection of documents. You could also accomplish this manually by using specialized data readers for each file format in your collection of data. However, using **LlamaParse** will greatly simplify this process, improve the overall quality, and save you a lot of time.

**LlamaParse** supports a wide and expanding range of file types beyond just PDFs, including Word docs, PowerPoint, RTF, ePub, and many more. It offers a generous free tier to get started.

The necessary **LlamaParse** integration package should already be installed along with the LlamaIndex components, so no additional installation is required to run the code example in this section.

The next step is to create a free account on https://cloud.llamaindex.ai and obtain an API key. Once you have obtained the key, you can use it directly in your code, but for a more secure approach, I strongly encourage you to follow the same steps we followed in *Chapter 2*, *LlamaIndex: The Hidden Jewel - An Introduction to the LlamaIndex Ecosystem*, and add the key as a variable in your local environment under the name **LLAMA_CLOUD_API_KEY**. To demonstrate the capabilities of this tool, I've designed a sample PDF with a more complex structure, as can be seen in Figure 4.1:

![Figure 4.1 – A sample PDF containing multiple articles, images, and tables](images/chapter-04-figure-04-1-sample-pdf-multiple-articles-images-tables.jpg)

*Figure 4.1 – A sample PDF containing multiple articles, images, and tables*

Here's a basic code example that uses **LlamaParse** to ingest this PDF:

```python
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
```

The first part of the code imported the necessary modules. Next, we'll configure **LlamaParse** and pass it to **SimpleDirectoryReader**:

```python
parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="Extract all text content, preserve table structure, and format code snippets in markdown"
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./files",
    file_extractor=file_extractor
).load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What are the main topics covered in this document?")
print(response)
```

In this example, we've configured **LlamaParse** to output results in markdown format and provided specific parsing instructions. The **file_extractor** parameter tells **SimpleDirectoryReader** to use our custom parser for PDF files while using default parsers for other file types.

The beauty of this approach is that **LlamaParse** handles the complex parsing while **SimpleDirectoryReader** manages the file discovery and coordination. This combination gives you the best of both worlds: advanced parsing capabilities with simple bulk ingestion.

## Parsing the documents into nodes

Once we have our documents loaded, the next step is to parse them into nodes. As we learned in the previous chapter, nodes are smaller, more manageable chunks of content that can be efficiently processed and indexed.

LlamaIndex provides several node parsers, each designed for different use cases and document types. Let's explore the most commonly used ones.

### Text splitters and node parsers

The most fundamental way to create nodes is by splitting documents into smaller text chunks. LlamaIndex offers several text splitters:

#### TokenTextSplitter

This is the most commonly used splitter. It breaks text into chunks based on token count, ensuring that each chunk fits within the context window of your LLM.

```python
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document

text = "This is a long document that needs to be split into smaller chunks for processing..."
doc = Document(text=text)

splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separator=" "
)

nodes = splitter.get_nodes_from_documents([doc])
for node in nodes:
    print(f"Node: {node.text[:100]}...")
```

Key parameters:
- **chunk_size**: Maximum number of tokens per chunk
- **chunk_overlap**: Number of tokens to overlap between chunks (helps maintain context)
- **separator**: Character or string used to split text

#### SentenceSplitter

This splitter attempts to break text at sentence boundaries, creating more semantically coherent chunks.

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20
)

nodes = splitter.get_nodes_from_documents(documents)
```

#### CodeSplitter

For code documents, this specialized splitter understands programming language syntax and splits code at logical boundaries.

```python
from llama_index.core.node_parser import CodeSplitter

splitter = CodeSplitter(
    language="python",
    chunk_lines=40,
    chunk_lines_overlap=15,
    max_chars=1500
)

nodes = splitter.get_nodes_from_documents(code_documents)
```

### Hierarchical node parsing

For documents with clear hierarchical structure (like books with chapters and sections), you can use hierarchical parsing to maintain the document structure.

```python
from llama_index.core.node_parser import HierarchicalNodeParser

parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = parser.get_nodes_from_documents(documents)
```

This creates nodes at multiple levels of granularity, allowing for both broad context and specific details during retrieval.

### Semantic splitters

These advanced splitters use embeddings to determine optimal split points based on semantic similarity.

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=embed_model,
)

nodes = splitter.get_nodes_from_documents(documents)
```

This approach creates more coherent chunks by identifying natural breakpoints in the content based on semantic similarity.

## Working with metadata to improve the context

Metadata is crucial for improving the quality and relevance of your RAG system. It provides additional context that can be used for filtering, routing, and enhancing retrieval accuracy.

### Manual metadata addition

You can manually add metadata when creating documents:

```python
from llama_index.core import Document

doc = Document(
    text="This is the content of the document...",
    metadata={
        "title": "Introduction to Machine Learning",
        "author": "John Doe",
        "category": "education",
        "difficulty": "beginner",
        "last_updated": "2024-01-15"
    }
)
```

### Automatic metadata extraction

LlamaIndex provides several extractors that can automatically generate metadata:

#### TitleExtractor

Extracts potential titles from document content:

```python
from llama_index.core.extractors import TitleExtractor

title_extractor = TitleExtractor(nodes=5)
```

#### SummaryExtractor

Generates summaries of document chunks:

```python
from llama_index.core.extractors import SummaryExtractor

summary_extractor = SummaryExtractor(
    summaries=["prev", "self", "next"],
    prompt_template="Summarize the following text in 2-3 sentences: {context_str}"
)
```

#### KeywordExtractor

Extracts important keywords from content:

```python
from llama_index.core.extractors import KeywordExtractor

keyword_extractor = KeywordExtractor(keywords=10)
```

#### QuestionsAnsweredExtractor

Generates questions that the content can answer:

```python
from llama_index.core.extractors import QuestionsAnsweredExtractor

qa_extractor = QuestionsAnsweredExtractor(questions=3)
```

### Using metadata extractors in practice

Here's how to combine multiple extractors:

```python
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=512, chunk_overlap=50),
        TitleExtractor(nodes=5),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(keywords=5),
    ]
)

nodes = pipeline.run(documents=documents)

# View the extracted metadata
for node in nodes[:3]:
    print(f"Text: {node.text[:100]}...")
    print(f"Metadata: {node.metadata}")
    print("---")

### Entity extraction

For more advanced use cases, you can extract named entities from your content:

```python
from llama_index.core.extractors import EntityExtractor

entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # return entity names only
    device="cpu",  # or "cuda" if you have GPU
)
```

This extractor uses a pre-trained NER (Named Entity Recognition) model to identify people, organizations, locations, and other entities in your text.

### Custom metadata extractors

You can also create custom extractors for domain-specific metadata:

```python
from llama_index.core.extractors.metadata_extractors import MetadataExtractor
from llama_index.core.schema import BaseNode
from typing import List, Dict, Any

class CustomDateExtractor(MetadataExtractor):
    def extract(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        metadata_list = []
        for node in nodes:
            # Custom logic to extract dates from text
            import re
            date_pattern = r'\d{4}-\d{2}-\d{2}'
            dates = re.findall(date_pattern, node.text)
            metadata_list.append({"extracted_dates": dates})
        return metadata_list

custom_extractor = CustomDateExtractor()
```

## Estimating the potential cost of using metadata extractors

When using LLM-based metadata extractors, it's important to understand the cost implications. Each extractor that uses an LLM (like SummaryExtractor, QuestionsAnsweredExtractor) will make API calls to your chosen LLM provider.

### Cost calculation example

Let's estimate the cost for processing a document collection:

```python
# Example calculation for OpenAI GPT-3.5-turbo
# Assuming:
# - 100 documents
# - Average 2000 tokens per document
# - 3 LLM-based extractors (Summary, Keywords, Q&A)
# - GPT-3.5-turbo pricing: $0.0015 per 1K input tokens, $0.002 per 1K output tokens

total_documents = 100
avg_tokens_per_doc = 2000
num_llm_extractors = 3

# Input tokens (document content sent to LLM)
total_input_tokens = total_documents * avg_tokens_per_doc * num_llm_extractors
input_cost = (total_input_tokens / 1000) * 0.0015

# Output tokens (generated metadata, estimated)
avg_output_tokens_per_extraction = 100
total_output_tokens = total_documents * avg_output_tokens_per_extraction * num_llm_extractors
output_cost = (total_output_tokens / 1000) * 0.002

total_cost = input_cost + output_cost
print(f"Estimated cost: ${total_cost:.2f}")
```

### Cost optimization strategies

1. **Use caching**: LlamaIndex's ingestion pipeline supports caching to avoid reprocessing unchanged content.

2. **Selective extraction**: Only use LLM-based extractors where they add significant value.

3. **Batch processing**: Process documents in batches to optimize API usage.

4. **Local models**: Consider using local models for some extractors to reduce API costs.

```python
from llama_index.llms.huggingface import HuggingFaceLLM

# Use a local model for summary extraction
local_llm = HuggingFaceLLM(
    model_name="microsoft/DialoGPT-medium",
    tokenizer_name="microsoft/DialoGPT-medium",
    device_map="auto",
)

summary_extractor = SummaryExtractor(
    summaries=["self"],
    llm=local_llm
)
```

## Preserving privacy with metadata extractors, and not only

When working with sensitive data, privacy considerations are paramount. Here are strategies to protect your data while still benefiting from metadata extraction:

### Using local models

Replace cloud-based LLMs with local alternatives:

```python
from llama_index.llms.ollama import Ollama

# Use Ollama for local LLM inference
local_llm = Ollama(model="llama2", request_timeout=60.0)

# Configure extractors to use local LLM
summary_extractor = SummaryExtractor(
    summaries=["self"],
    llm=local_llm
)
```

### Data anonymization

Anonymize sensitive information before processing:

```python
import re

def anonymize_text(text):
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # Replace phone numbers
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)

    # Replace SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    return text

# Apply anonymization before creating documents
anonymized_text = anonymize_text(original_text)
doc = Document(text=anonymized_text)
```

### Selective processing

Process only non-sensitive parts of documents:

```python
def filter_sensitive_content(text):
    # Remove sections marked as confidential
    lines = text.split('\n')
    filtered_lines = []
    skip_section = False

    for line in lines:
        if 'CONFIDENTIAL' in line.upper():
            skip_section = True
        elif 'END CONFIDENTIAL' in line.upper():
            skip_section = False
            continue

        if not skip_section:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)
```

## Using the ingestion pipeline to increase efficiency

The ingestion pipeline is a powerful feature that streamlines the entire data processing workflow while providing caching and optimization benefits.

### Basic pipeline setup

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor, KeywordExtractor
from llama_index.embeddings.openai import OpenAIEmbedding

# Create pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=512, chunk_overlap=50),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(keywords=5),
        OpenAIEmbedding(),
    ]
)

# Process documents
nodes = pipeline.run(documents=documents)
```

### Pipeline with caching

Caching prevents reprocessing of unchanged content:

```python
# Set up caching
cache = IngestionCache()

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=512, chunk_overlap=50),
        SummaryExtractor(summaries=["self"]),
        OpenAIEmbedding(),
    ],
    cache=cache
)

# First run - processes all documents
nodes = pipeline.run(documents=documents)

# Save cache for future use
cache.persist("./cache/ingestion_cache.json")

# Later, load cache and only process new/changed documents
cache = IngestionCache.from_persist_path("./cache/ingestion_cache.json")
pipeline = IngestionPipeline(
    transformations=[...],
    cache=cache
)
```

### Parallel processing

For large document collections, enable parallel processing:

```python
pipeline = IngestionPipeline(
    transformations=[...],
    num_workers=4,  # Process documents in parallel
)
```

## Handling documents that contain a mix of text and tabular data

Many real-world documents contain both text and structured data like tables. LlamaIndex provides specialized tools for handling these mixed-content documents.

### UnstructuredElementNodeParser

This parser can separate text and table elements:

```python
from llama_index.core.node_parser import UnstructuredElementNodeParser

parser = UnstructuredElementNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# Separate text and table nodes
text_nodes = []
table_nodes = []

for node in nodes:
    if "table" in node.metadata.get("category", "").lower():
        table_nodes.append(node)
    else:
        text_nodes.append(node)

print(f"Text nodes: {len(text_nodes)}")
print(f"Table nodes: {len(table_nodes)}")
```

The parser creates different types of nodes:
- **Text nodes**: Containing the text chunks
- **Table nodes**: Containing the table data and metadata, such as coordinates

Storing these elements as separate nodes allows more modular and meaningful processing later in the RAG workflow. The text can be indexed and searched normally with elements like keywords. The tables can be loaded into a **pandas DataFrame** or any structured database for SQL-based access. So, in complex cases involving mixed data types, leveraging **UnstructuredElementNodeParser** before ingestion enables better data organization.

You can find a complete demo for using **UnstructuredElementNodeParser** in the official LlamaIndex documentation: https://docs.llamaindex.ai/en/stable/examples/query_engine/sec_tables/tesla_10q_table.html.

With these new concepts in our toolbox, let's continue building our tutoring project.

## Hands-on – ingesting study materials into our PITS

It's time for some practice. We now have everything we need to continue building our project. Let's write the **document_uploader.py** module.

This module will take care of ingesting and preparing our available study material. The user can upload any available books, technical documentation, or existing articles to provide more context to our tutor.

First, we have the imports:

```python
from global_settings import STORAGE_PATH, CACHE_FILE
from logging_functions import log_action
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import SummaryExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
```

Next, we must define the main function that's responsible for handling the ingestion process. You'll notice that it uses an ingestion pipeline to both streamline the code but also benefit from caching:

```python
def ingest_documents():
    documents = SimpleDirectoryReader(
        STORAGE_PATH,
        filename_as_id=True
    ).load_data()

    for doc in documents:
        print(doc.id_)
        log_action(
            f"File '{doc.id_}' uploaded by user",
            action_type="UPLOAD"
        )
```

The function loads all readable documents available in **STORAGE_PATH**, which was defined in **global_settings.py**.

For each document processed, a new event is stored in our log file using **log_action** from **logging_functions.py**.

Next, the function checks if there's already cached pipeline data to use:

```python
    try:
        cached_hashes = IngestionCache.from_persist_path(
            CACHE_FILE
        )
        print("Cache file found. Running using cache...")
    except:
        cached_hashes = ""
        print("No cache file found. Running without...")
```

The next step is to define and run the pipeline. If hashes from the cache file correspond, no operations should be processed; instead, the values will be directly loaded from the cache:

```python
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(
                chunk_size=1024,
                chunk_overlap=20
            ),
            SummaryExtractor(summaries=['self']),
            OpenAIEmbedding()
        ],
        cache=cached_hashes
    )

    nodes = pipeline.run(documents=documents)
    pipeline.cache.persist(CACHE_FILE)
    return nodes
```

We run three transformations in the pipeline:

- Basic chunking using **TokenTextSplitter**.
- A metadata extractor that summarizes each node.
- Embedding generation using **OpenAIEmbedding**. Don't worry about this step for now. I will explain it thoroughly in *Chapter 5*, *Indexing with LlamaIndex*.

In the end, the function saves the current data in the cache file and returns the processed nodes.

That's it for now. We have now uploaded and prepared the study materials for future processing. We'll continue with the indexing part in the next chapter.

## Summary

LlamaHub offers a variety of pre-built data loaders, streamlining the process of importing data from various sources as documents. This eliminates the need for creating unique parsers for different data formats.

After data is imported, it undergoes further processing into nodes, and we discussed various customization options available.

There's a broad range of options for metadata extraction, and the parsing process can be tailored to meet specific requirements.

Developing pipelines for data ingestion is an invaluable tool for enhancing the efficiency, both in terms of cost and time, of our RAG applications. It's also vital to keep privacy considerations in mind.

With data ingestion completed, let's continue our journey and discover the indexing powers of LlamaIndex.
```
