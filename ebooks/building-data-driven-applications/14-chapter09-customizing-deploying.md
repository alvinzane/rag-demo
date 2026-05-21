# Chapter 9: Customizing and Deploying Our LlamaIndex Project

Customizing **Retrieval-Augmented Generation** (**RAG**) components and optimizing performance is critical to building robust, production-ready applications with LlamaIndex. This chapter explores methods for leveraging open source models, intelligent routing across **large language models** (**LLMs**), and using community-built modules to increase flexibility and cost-effectiveness. Advanced tracing, evaluation methods, and deployment options are explored to gain deep insight, ensure reliable operation, and streamline the development life cycle.

Throughout this chapter, we're going to cover the following main topics:

- Customizing our RAG components
- Using advanced tracing and evaluation techniques
- Introduction to deployment with Streamlit
- Hands-on – a step-by-step deployment guide

## Technical requirements

For this chapter, you will need to install the following package in your environment:

- **Arize AI Phoenix**: https://pypi.org/project/arize-phoenix/

Three additional integration packages are required in order to run the sample code:

- **Hugging Face embeddings**: https://pypi.org/project/llama-index-embeddings-huggingface/
- **Zephyr query engine**: https://pypi.org/project/llama-index-packs-zephyr-query-engine/
- **Neutrino LLM**: https://pypi.org/project/llama-index-llms-neutrino/

All code samples from this chapter can be found in the **ch9** subfolder of the book's GitHub repository:

https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex

## Customizing our RAG components

For starters, let's talk about which components of a RAG workflow can be customized in LlamaIndex. The short answer is *pretty much all of them, as we have seen already in the previous chapters*. The fact that the framework itself is flexible and allows customization of all the core components is a definite advantage. But leaving aside the framework itself, the core of a RAG workflow is actually the LLM and the embedding model it uses. In all the examples given so far, we have used the default configuration of LlamaIndex – which is based on OpenAI models. But, as we already briefly discussed in *Chapter 3*, *Kickstarting Your Journey with LlamaIndex*, there are both good reasons and enough options available to choose other models – both commercial variants offered by established companies in this market, and open source models, which can be hosted locally, offering private alternatives, and substantially reducing the costs of a large-scale implementation.

But first, some background.

## How LLaMA and LLaMA 2 changed the open source landscape

In early 2023, Meta AI introduced the **Large Language Model Meta AI** (**LLaMA**) family, offering a notable leap in accessibility for LLMs by releasing model weights to the research community. Following this, LLaMA 2 was launched in July 2023, with improvements such as increased data for training and expanded model sizes, alongside models fine-tuned for dialogue under less restrictive commercial use conditions. Meta developed and launched three versions of LLaMA 2 with 7, 13, and 70 billion parameters, respectively. While the basic structure of these models stayed similar to the original LLaMA versions, they were trained with 40% additional data compared to the original models, in order to enhance their foundational capabilities.

Despite some controversy regarding its open source status, the initiative marked a significant contribution to the open source ecosystem, triggering a new wave of community-based research and application development. The model consistently showcased competitive performance in tests against other leading LLMs, proving its advanced capabilities.

Further down the line, these releases have led to the creation of tools such as *llama.cpp* by Georgi Gerganov (https://github.com/ggerganov/llama.cpp), enabling the operation of these sophisticated models on more modest hardware, thus democratizing access to cutting-edge AI technologies.

**Quick note**

*llama.cpp* is an efficient C/C++ implementation of Meta's LLaMA architecture for LLM inference. Hugely popular in the open source community, with more than 43,000 stars on GitHub and over 930 releases, this foundational framework has sparked the development of many other similar tools and services such as Ollama, Local.AI, and others. These updates and advances signaled that AI research was changing, focusing more on making information freely available and making sure AI models can run on simpler computers and other edge devices. This opened up more possibilities for using **generative AI** (**GenAI**) and encouraged new ideas and improvements everywhere.

I won't go into a detailed discussion of all the currently available tools for running local LLMs. This is because there is already a plethora of available methods by which various open source models can be run on the local system. And not just local LLMs: there's also an increasing number of service providers offering access either to their own proprietary AI models or providing cloud-hosted access to open source models, and the good news is that LlamaIndex already provides built-in support for many of them. You can always consult the official documentation of the framework for a detailed overview of the supported models, along with examples of how they can be used: https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules.html.

Instead, I will try to offer you an alternative that I personally find very convenient for two important reasons: it is very easy to implement, and your existing code can be reused with only a few minimal changes. For beginner coders and tinkerers wanting to quickly experiment with an idea or build simple prototypes, this may be one of the best solutions.

## Running a local LLM using LM Studio

Built on top of the **llama.cpp** library, **LM Studio** (https://lmstudio.ai/) provides a very user-friendly graphical interface for LLMs. It allows us to download, configure, and locally run almost any open source model available on Hugging Face. A great resource, especially for non-technical users, LM Studio offers two ways of interacting with a local LLM: through a chat UI similar to OpenAI's ChatGPT or via an OpenAI-compatible local server. This second option makes it particularly useful because we can easily adapt any LlamaIndex application natively designed to use OpenAI's LLMs with very few modifications. We'll get to that in a moment, but first, let's see how to get things started with LM Studio.

To start using this tool, you'll first have to download and install the right version, depending on your operating system. Releases are available for Mac, Windows, and Linux. The installation steps are self-explanatory and well documented on their website.

Once installed, the LM Studio GUI starts with a **Model Discovery** screen where you can type any model or model family name and get a list of matching model builds available for download. We'll use the popular **Zephyr-7B** model for our example (https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). I have specifically chosen Zephyr because, albeit a compact model, it demonstrates the effectiveness of distilling an LLM into a more manageable size. Derived from **Mistral-7B**, Zephyr-7B establishes a new benchmark for chat models with 7 billion parameters, surpassing the performance of **LLAMA2-CHAT-70B** on the Hugging Face *LMSYS Chatbot Arena Leaderboard* (https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard). *Figure 9.1* shows a typical output when searching for the **zephyr-7b** keyword:

![Figure 9.1 – LM Studio screenshot displaying search results](images/chapter-09-figure-09-1-lm-studio-screenshot-displaying-search-results.jpg)

*Figure 9.1 – LM Studio screenshot displaying search results*

In the search results screen, you'll see two panels:

- The one on the left contains all models that match your search query. In our case, these are different builds of the Zephyr-7B model
- The right panel lists all the **Generative Pre-trained Transformer-Generated Unified Format** (**GGUF**) file versions available for download

**About GGUF files**

GGUF is a specific file format used for storing models for inference. Enhancing model sharing and usage efficiency, this format has quickly become a popular way of storing and distributing models throughout the open source community.

For most models, you'll get an entire list of GGUF files available. Each one will have its own characteristics, but probably the most important characteristic is the **quantization** level.

### Understanding LLM quantization

Running an open source LLM on typical consumer hardware can prove challenging mainly because of its large memory footprint and high computational requirements. While some consumer-grade GPUs can aid in this regard, they may not be as effective as enterprise-level hardware in handling the demands of LLMs. That's why we need quantization. The goal of applying quantization – a post-training optimization technique – to an AI model is to optimize it for better performance and efficiency, particularly in terms of speed and memory usage, without significantly compromising its accuracy or output quality.

The quantization process achieves this by converting the model's parameters – typically stored as 32-bit floating-point numbers – to lower-bit representations, such as **16-bit floating-point** (**FP16**), **8-bit integers** (**INT8**), or even lower. It's a kind of approximation process that works by reducing the numerical precision used to represent the model's parameters, combined with complex techniques to maintain as much accuracy as possible. Modern quantization techniques are designed to minimize accuracy loss, often resulting in models that are nearly as accurate as their full-precision counterparts.

**A simple analogy to help you better understand the concept**

Imagine you have a recipe that calls for very precise measurements, such as **"1.4732"** cups of flour. In practice, you might round this to 1.5 cups, as the difference is negligible in most cases and the difference will not affect the end result. This is similar to quantization, where we reduce the precision of the model's parameters to make the model more efficient while maintaining acceptable accuracy. But instead of cups of flour, we reduce the numerical precision of the model's parameters. Instead of using 16 bits to store a parameter as 23.7, we could quantize it into 8 bits as 23. This directly translates to less memory usage and faster processing times. However, there is a trade-off between model size, speed, and accuracy.

With an acceptable loss of accuracy, this process can significantly reduce the size of the model and the computational resources required for both training and inference phases, making it more feasible to deploy these models on consumer hardware. Generally, the lower the bit representation (such as **INT4** or even binary), the smaller and faster the model becomes, but at a higher risk of accuracy loss.

Being built on top of llama.cpp, LM Studio can take advantage of any compatible GPUs that could be used during the inference process. This feature is commonly called *GPU offloading* and means that computing operations can be partially or even entirely transferred from the CPU to the GPU. Given the fact that a modern GPU is capable of handling highly parallel computing tasks more efficiently than CPUs, this can dramatically speed up the inference process. It also reduces the load on the CPU, thus providing an overall balanced improvement of system performance. The main limitation when attempting GPU offloading is the amount of video memory available on your GPU. In order to run efficiently, the GPU must load the model in the video memory first.

Because of this, apart from the quantization level, the GGUF files in the right panel will also have a flag showing three possible compatibility scenarios, each represented by a different color:

- **Green**: This means your GPU has enough video memory to load the model and execute the inference. In most cases, this is the ideal scenario
- **Blue**: Not ideal, but still provides a considerable uplift in performance
- **Gray**: This may or may not work depending on the model architecture
- **Red**: Unfortunately, this means you won't be able to run this version on your machine, the most probable reason being that its size exceeds your total system memory

**Pro tip**

A very handy tool for approximating the required VRAM for a particular model given a particular quantization level can be found on the Hugging Face website: https://huggingface.co/spaces/hf-accelerate/model-memory-usage

### So, which model should you choose?

The general rule of thumb is that with a lower quantization level, less memory will be required and the inference process will be faster. The trade-off is decreased accuracy. For example, a 3-bit quantization will always result in less accuracy than a 6-bit quantization.

Once you've made a decision on the exact model version, the next step is to download the model on your machine. But first, make sure you have the necessary space on your hard drive. There's a handy status bar on the bottom of the UI to monitor the status of the download.

After the download is complete, moving to the **Chats** screen will display something similar to *Figure 9.2*:

![Figure 9.2 – LM Studio's chat UI](images/chapter-09-figure-09-2-lm-studio-chat-ui.jpg)

*Figure 9.2 – LM Studio's chat UI*

This is the interaction method that I mentioned at the beginning of this section – the one resembling the ChatGPT interface. In this screen, you'll be able to do the following:

- Select the desired AI model from a list of all downloaded ones. To choose your model, use the *model selector* on top of the screen. You'll have to wait for a few moments until the model is loaded into memory.
- Configure any available parameters of the model using the *configuration panel* on the right side. We'll talk in more detail about that in a moment.
- See a list of previous chats on the left side.
- Chat with the model using a familiar interface inspired by ChatGPT.

There are a number of parameters that you can tweak in the configuration panel. The most important ones are the following:

- **Preset**: Some models come with predefined configurations that you can load from presets. For an easy start, I would recommend selecting the model's specific preset from the list. For example, there is a Zephyr preset that can be used with all Zephyr-based models
- **System Prompt**: This prompt will set the initial context of the conversation
- **GPU Offload**: Allows you to configure the number of model layers to be offloaded to the GPU. Depending on the model you're using and your available GPU, you may want to gradually experiment with increasing values while checking for model stability. Higher values can sometimes produce errors. If you feel confident, use -1 to offload all the model's layers to the GPU
- **Context Length**: Allows you to define the maximum context window to be used

Changing some of these parameters may trigger a model reload, so you'll have to be patient until it completes the process. Once you have customized everything, the floor is yours – enjoy chatting with your local LLM.

### So far, so good, but where's the RAG part in all this?

For that, we'll have to go to the **Local Inference Server** screen, which you can do by pressing the double-arrow icon on the left-side menu. You'll be presented with a UI similar to *Figure 9.3*:

![Figure 9.3 – The local Inference Server interface in LM Studio](images/chapter-09-figure-09-3-local-inference-server-interface-lm-studio.jpg)

*Figure 9.3 – The local Inference Server interface in LM Studio*

The configuration options from the right-side panel are almost identical to the ones in the **Chat** screen. In the beginning, you can leave the *server configuration* options as default. The *usage* section tells you how to interact with the API. One of the great aspects of LM Studio is that it emulates the OpenAI API. That means your already existing code will need very few changes to work with a local LLM hosted through LM Studio.

All you have to do at this point is to click the **Start Server** button, and you're good to go.

**Quick note**

Please keep in mind that while the API server is running, the chat UI will be disabled, so you won't be able to use both at the same time.

Let's see exactly what we need to change in our code if we want to port it to a local LLM using this method. If we look at the recommendation in the *usage* section, we'll see that a single change is necessary:

```python
client = OpenAI(base_url="http://localhost:1234/v1")
```

However, because LlamaIndex has its own implementation of the OpenAI API client, in our case, we'll have to use the **api_base** parameter like this:

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base='http://localhost:1234/v1',
    temperature=0.7
)

print(llm.complete('Who is Lionel Messi?'))
```

As you can see, the only real change we have to make is pointing the **llm** instance toward our local server instead of the OpenAI one. The rest of the code remains unchanged. After running this example, you'll see actual requests coming from our code and responses coming from the API in LM Studio's log screen. If you want to permanently reconfigure the LLM in the entire code, you'll have to define a **Settings** object and use it to configure global settings, as I showed you in *Chapter 3*, *Kickstarting Your Journey with LlamaIndex*, in the *Customizing the LLM used by LlamaIndex* section.

Neat, isn't it? Our data is now completely private, and we don't have to pay for using an AI model in our RAG workflows anymore. Of course, there's still a cost, albeit in electricity rather than tokens. The capability to run local models on modest hardware unlocks numerous possibilities that extend beyond mere text generation. This includes the opportunity to embrace fully multimodal experiences with models such as **LLaVa** (https://huggingface.co/docs/transformers/main/en/model_doc/llava), allowing for a wider range of applications: a wonderful tool that serves as an excellent resource for rapid prototyping or exploring diverse ideas.

However, keep in mind that LM Studio is governed by a licensing model, which restricts its use to personal, non-commercial purposes. To utilize LM Studio for commercial applications, obtaining permission from the developers is necessary.

## Routing between LLMs using services such as Neutrino or OpenRouter

Sometimes, a single LLM may not be ideal for every single interaction. In complex RAG scenarios, finding the best mix between cost, latency, and precision could prove to be a difficult task when forced to choose a single LLM for everything. But what if we could find a way to mix different LLMs in the same app and dynamically choose which one to use for each individual interaction? That is the exact purpose of third-party services such as **Neutrino** (https://www.neutrinoapp.com/) and **OpenRouter** (https://openrouter.ai/). These types of services can significantly enhance a RAG workflow by providing intelligent routing capabilities for queries across different LLMs.

Neutrino's smart model router, for example, automatically selects the most appropriate model for each query based on factors such as cost, speed, and quality requirements. This dynamic selection process ensures optimal performance while maintaining cost efficiency. The service supports a wide range of models, from lightweight options for simple queries to more powerful models for complex reasoning tasks.

Similarly, OpenRouter provides access to multiple AI models through a unified API, allowing developers to experiment with different models and automatically route queries to the most suitable one. Both services offer significant advantages:

- **Cost optimization**: Route simple queries to cheaper models and complex ones to premium models
- **Performance optimization**: Balance speed and accuracy based on query requirements
- **Fallback mechanisms**: Automatically switch to alternative models if the primary choice is unavailable
- **A/B testing**: Compare performance across different models for the same queries

Here's how you can integrate Neutrino with LlamaIndex:

```python
from llama_index.llms.neutrino import Neutrino

# Initialize Neutrino LLM
llm = Neutrino(
    api_key="your_neutrino_api_key",
    router="default"  # Uses Neutrino's intelligent routing
)

# Use with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What are the main topics in these documents?")
```

## Leveraging community-built Llama Packs

**Llama Packs** are pre-built, community-contributed modules that provide ready-to-use implementations of common RAG patterns and advanced techniques. These packs can significantly accelerate development by providing tested, optimized solutions for specific use cases.

Some popular Llama Packs include:

- **Zephyr Query Engine Pack**: Optimized query engine specifically designed for Zephyr models
- **Multi-Document Agents Pack**: Agents that can work across multiple document collections
- **Evaluation Pack**: Comprehensive evaluation tools for RAG systems
- **Dense X Retrieval Pack**: Advanced retrieval techniques for better context selection

Here's how to use the Zephyr Query Engine Pack:

```python
from llama_index.packs.zephyr_query_engine import ZephyrQueryEnginePack

# Download and initialize the pack
pack = ZephyrQueryEnginePack.from_defaults(
    documents=documents,
    llm_model="zephyr-7b-beta"
)

# Use the optimized query engine
response = pack.run("What are the key insights from the documents?")
```

## Using advanced tracing and evaluation techniques

Understanding how your RAG system performs and identifying bottlenecks is crucial for production deployments. LlamaIndex integrates with advanced tracing and evaluation tools to provide deep insights into your application's behavior.

### Phoenix Tracing

**Phoenix** by Arize AI provides comprehensive tracing capabilities for LlamaIndex applications. It allows you to visualize the entire execution flow, from query processing to response generation, helping identify performance bottlenecks and optimization opportunities.

To set up Phoenix tracing:

```python
import phoenix as px
from llama_index.core import set_global_handler

# Launch Phoenix
session = px.launch_app()

# Set Phoenix as the global handler
set_global_handler("arize_phoenix")

# Your LlamaIndex code here
# All operations will be automatically traced
```

Phoenix provides several key features:

- **Execution Flow Visualization**: See the complete path from query to response
- **Performance Metrics**: Track latency, token usage, and costs
- **Error Detection**: Identify and debug failures in your RAG pipeline
- **Comparative Analysis**: Compare different configurations and models

### Evaluation with Phoenix

Phoenix also provides built-in evaluators for assessing RAG system quality:

```python
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator
)

# Initialize evaluators
hallucination_eval = HallucinationEvaluator()
qa_eval = QAEvaluator()
relevance_eval = RelevanceEvaluator()

# Evaluate your responses
eval_results = []
for query, response, context in test_data:
    # Check for hallucinations
    hallucination_score = hallucination_eval.evaluate(
        input=query,
        output=response,
        reference=context
    )

    # Evaluate QA correctness
    qa_score = qa_eval.evaluate(
        input=query,
        output=response,
        reference=expected_answer
    )

    # Check relevance
    relevance_score = relevance_eval.evaluate(
        input=query,
        output=response
    )

    eval_results.append({
        'query': query,
        'hallucination': hallucination_score,
        'qa_correctness': qa_score,
        'relevance': relevance_score
    })
```

### Custom Evaluation Metrics

You can also create custom evaluation metrics tailored to your specific use case:

```python
from llama_index.core.evaluation import BaseEvaluator

class CustomRelevanceEvaluator(BaseEvaluator):
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, query, response, contexts=None, **kwargs):
        # Custom evaluation logic
        prompt = f"""
        Query: {query}
        Response: {response}

        Rate the relevance of the response to the query on a scale of 1-5.
        Provide only the numeric score.
        """

        result = self.llm.complete(prompt)
        try:
            score = float(result.text.strip())
            return score / 5.0  # Normalize to 0-1
        except:
            return 0.0

# Use custom evaluator
custom_eval = CustomRelevanceEvaluator(llm)
score = custom_eval.evaluate(query, response)
```

## Introduction to deployment with Streamlit

**Streamlit** is a powerful Python framework that makes it easy to create and deploy web applications for machine learning and data science projects. For RAG applications built with LlamaIndex, Streamlit provides an excellent platform for creating user-friendly interfaces and deploying applications to the cloud.

### Why Streamlit for RAG Applications?

Streamlit offers several advantages for RAG deployments:

- **Rapid Development**: Create web interfaces with minimal code
- **Interactive Components**: Built-in widgets for file uploads, chat interfaces, and data visualization
- **Easy Deployment**: Multiple deployment options including Streamlit Community Cloud
- **Real-time Updates**: Automatic reloading when code changes
- **Session State Management**: Handle user sessions and conversation history

### Basic Streamlit RAG Application

Here's a simple example of a Streamlit application for a RAG system:

```python
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.vector_store import SimpleVectorStore

st.title("RAG Application with LlamaIndex")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload
uploaded_files = st.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
    type=['txt', 'pdf', 'docx']
)

if uploaded_files and st.session_state.index is None:
    with st.spinner("Processing documents..."):
        # Save uploaded files temporarily
        docs = []
        for file in uploaded_files:
            # Process each file and create documents
            # This is simplified - you'd need proper file handling
            content = file.read().decode()
            docs.append(Document(text=content))

        # Create index
        st.session_state.index = VectorStoreIndex.from_documents(docs)
        st.success("Documents processed successfully!")

# Chat interface
if st.session_state.index:
    query = st.chat_input("Ask a question about your documents")

    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Get response from RAG system
        query_engine = st.session_state.index.as_query_engine()
        response = query_engine.query(query)

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": str(response)
        })

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
```

### Advanced Streamlit Features for RAG

Streamlit provides several advanced features that are particularly useful for RAG applications:

#### Sidebar Configuration

```python
# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Model selection
    model_choice = st.selectbox(
        "Choose LLM",
        ["gpt-3.5-turbo", "gpt-4", "local-llm"]
    )

    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Top-k setting
    top_k = st.number_input(
        "Top-k results",
        min_value=1,
        max_value=10,
        value=3
    )
```

#### Progress Tracking

```python
# Show progress during document processing
progress_bar = st.progress(0)
status_text = st.empty()

for i, doc in enumerate(documents):
    status_text.text(f'Processing document {i+1}/{len(documents)}')
    # Process document
    progress_bar.progress((i + 1) / len(documents))

status_text.text('Processing complete!')
```

#### Caching for Performance

```python
@st.cache_resource
def load_index():
    """Cache the index to avoid reloading on every interaction"""
    return VectorStoreIndex.from_documents(documents)

@st.cache_data
def get_response(query, _index):
    """Cache responses to avoid recomputing identical queries"""
    query_engine = _index.as_query_engine()
    return str(query_engine.query(query))

## Hands-on – a step-by-step deployment guide

Now let's walk through deploying our PITS (Personal Intelligent Tutoring System) application to Streamlit Community Cloud. This will make our RAG application accessible to users worldwide.

### Prerequisites

Before we begin, ensure you have:

1. A GitHub account
2. The PITS application code ready for deployment
3. All necessary dependencies listed in a `requirements.txt` file
4. OpenAI API key (or other LLM service credentials)

### Step 1: Preparing the Repository

The main file, **app.py**, which is the main file in our case, is currently found in the **Building-Data-Driven-Applications-with-LlamaIndex\PITS_APP** folder. To fix that, we'll first make a copy of the **PITS_APP** subfolder, and then we'll initiate a new GitHub repository from that new folder. To keep things simple and require minimum changes, I will guide you on how to create a new repository containing just the PITS app and then deploy it from your own GitHub account:

1. First, let's create a copy of our local **PITS_APP** subfolder. Open Command Prompt and navigate to the **Building-Data-Driven-Applications-with-LlamaIndex** folder of your cloned repository. From that folder, type the following command: **xcopy PITS_APP C:\PITS_APP /E /I**

2. This will create a folder on your **"C:"** drive containing only the source files of the PITS application. If you navigate to the newly created folder and list its contents with the **dir** command, the output should look like *Figure 9.8*:

![Figure 9.8 – The contents of the C:\PITS_APP folder](images/chapter-09-figure-09-8-contents-pits-app-folder.jpg)

*Figure 9.8 – The contents of the C:\PITS_APP folder*

3. The next step is to sign in to your GitHub account and create a new repository. Let's name it **PITS_ONLINE**, as in *Figure 9.9*:

![Figure 9.9 – Creating a new GitHub repository named PITS_ONLINE](images/chapter-09-figure-09-9-creating-new-github-repository-pits-online.jpg)

*Figure 9.9 – Creating a new GitHub repository named PITS_ONLINE*

4. Once created, note the repository URL for the next steps. Next, we'll initialize a new local repository in the desired folder. Open your CLI and navigate to the folder you want to turn into a separate repository – **C:\PITS_APP** – then execute the following command: **git init**

5. Next, add and commit the existing files by running the following command:

```bash
git add .
git commit -m "Initial commit for PITS_ONLINE repository"
```

6. It's now time to link your local repository to the GitHub repository you created. Replace the URL with your GitHub URL and append **.git** at the end in the following command: **git remote add origin <your_repository_URL>.git**

7. And finally, we push the contents to the new online repository with the following command:

```bash
git branch -M main
git push -u origin main
```

If everything went smoothly you should now have a brand-new GitHub repository containing the PITS source code.

### Step 2: Deploying to Streamlit Community Cloud

Let's handle the Community Cloud deployment next.

Deploying Streamlit applications into their Community Cloud environment is a fairly simple and straightforward process. To begin our deployment, the first step is to sign up for a free Streamlit account here: https://share.streamlit.io/signup. The best option is to use your GitHub account both for signing up and signing in to your Streamlit account. Once logged in, simply click on the **New app** button to begin the deployment process. You'll be taken to a screen similar to what you can see in *Figure 9.10*:

![Figure 9.10 – Deploying an application into Streamlit Community Cloud](images/chapter-09-figure-09-10-deploying-application-streamlit-community-cloud.jpg)

*Figure 9.10 – Deploying an application into Streamlit Community Cloud*

If you signed in to Streamlit using GitHub, you should already have the **PITS_ONLINE** repository listed as an option. Select it, then, under the **Main file path** field, change the default value to **app.py** and then click **Deploy**. From here, the Streamlit deployment service takes over and prepares the required environment for your application. This might take a while, but if you want to check on the progress, you can always expand the **Manage app** section on the bottom right of your screen. When everything is ready, the application should start automatically.

You can now ingest your existing training materials, have PITS generate slides and narrations about your desired study topic, and ask its chatbot any questions related to the contents.

**Important note**

Don't forget, you're using your own API key. To keep costs under control, you should first experiment on a limited scale by uploading some small training resources and always keeping an eye on the OpenAI API usage. The good news is that the majority of the cost is incurred during slides and narration generation. However, once that is completed, the resulting material is stored and reused in future sessions.

Simple, isn't it? Although offering an environment with limited resources, the Streamlit Community Cloud service makes it really easy to deploy simple apps and share quick prototypes. Your app is now online and can easily be shared with other users.

If anything went wrong, though, and you didn't manage to complete the deployment, head over to the official documentation, and look for a solution: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app. In the Streamlit documentation, you'll also find additional deployment options and configurations available that might be useful for your future projects.

### Step 3: Environment Variables and Secrets

For production deployments, you'll need to properly handle sensitive information like API keys. Streamlit Community Cloud provides a secrets management system:

1. In your Streamlit app dashboard, click on the settings gear icon
2. Navigate to the "Secrets" section
3. Add your environment variables in TOML format:

```toml
[secrets]
OPENAI_API_KEY = "your-api-key-here"
DATABASE_URL = "your-database-url"
```

4. Access these secrets in your code:

```python
import streamlit as st

# Access secrets
api_key = st.secrets["OPENAI_API_KEY"]
```

### Step 4: Monitoring and Maintenance

Once deployed, monitor your application's performance:

- **Resource Usage**: Check memory and CPU usage in the Streamlit dashboard
- **Error Logs**: Monitor the logs for any runtime errors
- **User Feedback**: Implement feedback mechanisms to improve the application
- **Updates**: Use GitHub integration for automatic deployments when you push changes

### Alternative Deployment Options

While Streamlit Community Cloud is great for prototypes, consider these alternatives for production:

- **Streamlit Cloud for Teams**: Enhanced features and resources
- **Docker Containers**: Deploy on any cloud provider
- **Heroku**: Easy deployment with git integration
- **AWS/GCP/Azure**: Full control over infrastructure
- **Kubernetes**: For scalable, enterprise deployments

## Summary

In this chapter, we explored customizing and enhancing RAG workflows with LlamaIndex. We covered techniques to leverage open source LLMs such as Zephyr using tools such as LM Studio, offering cost-effective and privacy-focused alternatives to commercial models. The chapter discussed intelligent routing across multiple LLMs with services such as Neutrino and OpenRouter for optimized performance. Community-built Llama Packs were highlighted as powerful ways to rapidly prototype and build advanced components, and the chapter introduced the Llama CLI for streamlining RAG development and deployment workflows.

We talked about advanced tracing with Phoenix, allowing us to gain deep insight into application execution flows and pinpoint problems through visualization. The evaluation of RAG systems was covered using Phoenix's relevance, hallucination, and QA correctness evaluators, ensuring the robust performance of our LlamaIndex apps. Streamlit's deployment options, especially the Community Cloud service for easy application sharing, simplified the deployment process. A step-by-step guide demonstrated how to deploy the PITS tutoring application to the cloud.

With a strong grasp of customization, evaluation, and deployment techniques, developers can now build production-ready, optimized RAG applications tailored to their unique requirements.

Our journey continues with an exploration of the role of prompt engineering in enhancing the effectiveness of GenAI within the LlamaIndex framework.
```
