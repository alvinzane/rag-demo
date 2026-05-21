# Chapter 8: Building Chatbots and Agents with LlamaIndex

As this ebook edition doesn't have fixed pagination, the page numbers below are hyperlinked for reference only, based on the printed edition of this book.

This chapter provides an in-depth look at implementing chatbots and intelligent agents using the capabilities of LlamaIndex. We will explore the various chat engine modes available, from simple chatbots to more advanced context-aware and question-condensing engines. Then, we'll dive into agent architectures, analyzing tools, reasoning loops, and parallel execution methods. You will gain practical knowledge so that you can build conversational interfaces powered by LLMs that can understand user needs and orchestrate responses or actions by utilizing tools and data sources.

Throughout this chapter, we're going to cover the following main topics:

- Understanding chatbots and agents
- Implementing agentic strategies in our apps
- Hands-on – implementing conversation tracking for PITS

## Technical requirements

The following LlamaIndex integration packages will be required for the sample code:

- **Database Tool**: https://pypi.org/project/llama-index-tools-database/
- **OpenAI Agent**: https://pypi.org/project/llama-index-agent-openai/
- **Wikipedia Reader**: https://pypi.org/search/?q=llama-index-readers-wikipedia
- **LLM Compiler Agent**: https://pypi.org/project/llama-index-packs-agents-llm-compiler/

All the code samples in this chapter can be found in the **ch8** subfolder of this book's GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

## Understanding chatbots and agents

In the modern business ecosystem, the role of **chatbot systems** is increasingly important. First appearing in the 1960s (https://en.wikipedia.org/wiki/ELIZA), chatbots have always fascinated both developers and technology users alike. *Figure 8.1* shows the user interface of one of these early systems:

![Figure 8.1 – The ELIZA chatbot interface](images/chapter-08-figure-08-1-eliza-chatbot-interface.jpg)

*Figure 8.1 – The ELIZA chatbot interface*

While these systems were rudimentary initially and seen as more of an experiment, with the advancement of NLP technologies, the experience they offer has become increasingly interesting and valuable to users.

**Chatbot-based support systems** offer today's consumers a self-service experience. For users, self-service support services have two major advantages over human support:

- They are available 24/7, even outside normal working hours
- The user does not have to *hold the line* to access them

Even if there is some reluctance to use these systems at first, once they discover these advantages, users soon get used to interacting with them.

Don't necessarily think of chatbots as a technology designed to replace human support and interaction entirely. Although they have made enormous progress in recent years, these technologies, while getting more and more advanced, still have their shortcomings.

Lacking real empathy and the human touch, even under ideal operating conditions, chatbot-based services are unlikely to replace human support completely. But that doesn't mean they aren't extremely valuable, both for organizations and their users.

Perhaps the greatest value they bring is when they work in a blended experience, where users can receive both human support and access to self-service platforms that are interfaced with chatbot technologies. Implemented strategically, these systems can vastly improve not only the support offered to end consumers but also the internal interactions between an organization's employees.

**ChatOps**, for example, is a model increasingly used by modern organizations (https://www.ibm.com/blog/benefits-of-chatops/).

**Definition**

ChatOps refers to the ability to integrate chat platforms with operational workflows, facilitating transparent collaboration among team members, processes, tools, and automated bots to enhance service dependability, accelerate recovery, and boost collaborative productivity.

Based on the idea of **conversation-driven collaboration**, the ChatOps model combines **DevOps** principles (https://en.wikipedia.org/wiki/DevOps) by simplifying and accelerating interactions between team members using chatbots.

Whether we use them for internal communication or in interactions with our users, chatbots can only be useful to the extent that they can solve real problems. This depends on how well they can understand the context of the interaction and how relevant the answers they provide are.

*Figure 8.2* provides a visual representation of the ChatOps model:

![Figure 8.2 – The ChatOps paradigm](images/chapter-08-figure-08-2-chatops-paradigm.jpg)

*Figure 8.2 – The ChatOps paradigm*

If, in the beginning, the main limitation of chatbots came from the *clumsy* way of interacting with the user, with the evolution of NLP technologies, the main shortcoming has become, more recently, the lack of integration with the organization's knowledge base.

After all, what good is a natural communication experience if the answers given by the system aren't useful in solving the user's requests?

This brings us to RAG.

By now, I think it has become obvious that without being connected to an organization's knowledge base, a chatbot can, at best, be considered a technology experiment. Even conversational engines based on powerful LLMs such as GPT-4 can, at best, provide generic answers that don't always address the specific problems of each organization. Perhaps worse, not being anchored in validated documentation, they can *hallucinate* very convincingly, creating unpleasant or even potentially dangerous experiences.

As you've probably guessed by now, LlamaIndex also offers RAG tools for implementing chatbot technologies. In this chapter, we will explore the options available to us and understand how we can implement very simple systems to advanced chatbot mechanisms.

But first, let's see how this functionality is built into LlamaIndex.

## Discovering ChatEngine

In the previous chapters, we saw how we can build a query engine to run queries based on our data. This mechanism allows us to integrate multiple types of indexes, retrievers, node postprocessors, and response synthesizers at the same time, thus being able to access our proprietary data in multiple ways. Unfortunately, the **QueryEngine** class does not provide any mechanism to keep the history of a conversation. That means each query is a separate interaction and there is no contextual memory to allow a true *conversation*.

For that purpose, however, we have **ChatEngine**. Unlike query engines, **ChatEngine** allows us to have an actual conversation, giving us both the context of our proprietary data and the history of the chat. To simplify this concept even further, imagine a **QueryEngine** class with memory.

In its simplest form, a chat engine can be initialized just as easily, based on an index:

```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Hi, how are you?")
```

Once initialized, a chat engine can be queried using various methods:

- **chat()**: This method initiates a synchronous chat session, processing the user's message and returning the response immediately.
- **achat()**: This method is similar to **chat()** but executes the query asynchronously, allowing multiple requests to be processed simultaneously. This can be useful, for example, in a web or mobile application where we want to avoid blocking the main thread during server queries.
- **stream_chat()**: This method opens a streaming chat session, where responses can be returned as they are generated, for more dynamic interaction. This is particularly useful for long or complex responses that require significant processing time, allowing the user to start seeing parts of the response before all processing is complete.
- **astream_chat()**: This method is an asynchronous version of **stream_chat()** that allows us to handle streaming interactions in an asynchronous context.

Another option is to initiate a **Read-Eval-Print** (**REPL**) loop with **ChatEngine**:

```python
chat_engine.chat_repl()
```

A REPL chat is akin to a ChatGPT interface, where a user sends a message or question, the LLM processes the input, generates a response, and then immediately displays it to the user. This loop continues for as long as the user keeps providing input, creating an interactive conversation.

To reset a chat conversation, you can use the following command:

```python
chat_engine.reset()
```

This is useful when you want to clear the history and begin a new conversation thread.

So, the basics are very straightforward. Next, let's talk about the different **built-in chat modes** available in LlamaIndex.

## Understanding the different chat modes

When initializing a chat engine, we can use the **chat_mode** argument to invoke various chat engine types predefined in LlamaIndex. I will show you how each of these engines works. We will discuss them one by one and get a good understanding of the advantages and use cases best suited for each of them.

But first, let's have a short introduction to how chat memory is managed within LlamaIndex.

### Understanding how chat memory works

The **ChatMemoryBuffer** class is a specialized memory buffer that's designed to store chat history efficiently while also managing the token limit imposed by different LLMs. This structure is important because we can pass it as an argument when initializing chat engines using the **memory** parameter. By saving and restoring this buffer from one session to another, we can implement persistence for our conversations.

There are two different storage options for the chat store:

- The default **SimpleChatStore**, which stores the conversation in memory
- The more advanced **RedisChatStore**, which stores the chat history in a Redis database, eliminating the need to manually persist and load the chat history

The **chat_store** attribute, which is an instance of the **BaseChatStore** class, is used for the actual storage and retrieval of chat messages. This modular approach allows different storage implementations, such as a simple in-memory store or more complex database-backed stores.

We also have the **chat_store_key** parameter, which is used to uniquely identify the chat session or conversation within the chat store. This is useful for retrieving the correct conversation history when there are multiple conversations stored in the same chat store. Here's a basic example of **conversation history persistence** using **SimpleChatStore**:

```python
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
```

After importing the necessary libraries, we can try to load the previous conversation. If there is no previous conversation save file, we simply initialize an empty **chat_store**:

```python
try:
    chat_store = SimpleChatStore.from_persist_path(
        persist_path="chat_memory.json"
    )
except FileNotFoundError:
    chat_store = SimpleChatStore()
```

It's now time to initialize our memory buffer by using **chat_store** as an argument. Although not needed here, for a more detailed illustration, we will also customize **token_limit** and **chat_store_key**:

```python
memory = ChatMemoryBuffer.from_defaults(
    token_limit=2000,
    chat_store=chat_store,
    chat_store_key="user_X"
)
```

OK; we have all the necessary pieces. Let's put them together into a **SimpleChatEngine** class and create a chat loop:

```python
chat_engine = SimpleChatEngine.from_defaults(memory=memory)

while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Exiting chat...")
        break
    
    response = chat_engine.chat(user_message)
    print(f"Chatbot: {response}")
```

Once the user types **exit** and we break the loop, we use the **persist()** method to store the current conversation for future sessions:

```python
chat_store.persist(persist_path="chat_memory.json")
```

In case you're wondering why we haven't used the **chat_repl()** method shown previously and created a chat loop instead, the answer is in the following note.

**Important note**

While the **chat()**, **achat()**, **stream_chat()**, and **astream_chat()** methods can benefit from loading and resuming previous conversations, by design, the **chat_repl()** method will reset the conversation history during initialization.

**ChatMemoryBuffer** also plays an important role in ensuring that the conversation's context remains within the token limits of the model being used. Among other parameters available for **ChatMemoryBuffer**, the **token_limit** attribute specifies the maximum number of tokens that can be stored in the memory buffer. This limit is essential to ensure we stay within the maximum context window size of the current LLM we are using.

When the conversation exceeds the context limit, a sliding window method is applied. Older parts of the conversation are truncated to ensure that the most recent and relevant parts are retained and processed by the LLM within its token constraints.

**An analogy to better understand the sliding window method**

Imagine a conversation with an LLM as a train journey, where each piece of dialogue adds a carriage. However, the train can only be so long due to the tracks' length limit, representing the model's context window limit. To keep the journey going and add new carriages – in our case, messages – older ones need to be detached and left behind. This ensures the train can continue its journey, carrying the most recent and relevant parts of the conversation, while staying within the limits of the track. Just like in a train journey, where we might prioritize which carriages to keep based on their importance, the sliding window method prioritizes newer conversation parts, keeping the dialogue flowing smoothly.

Now that we understand how memory works, let's talk about the different available chat modes.

### Simple mode

This is the most **basic chat engine** available. It allows for a simple, direct conversation with the LLM, without any connection to our proprietary data. *Figure 8.3* explains this chat mode visually:

![Figure 8.3 – SimpleChatEngine](images/chapter-08-figure-08-3-simplechatengine.jpg)

*Figure 8.3 – SimpleChatEngine*

The user's experience in this mode is defined by the inherent capabilities and limitations of the LLM, such as its context window size and overall performance.

To initialize this mode, we can use the following code:

```python
from llama_index.core.chat_engine import SimpleChatEngine

chat_engine = SimpleChatEngine.from_defaults()
chat_engine.chat_repl()
```

If we want, we can customize the LLM using the **llm** argument:

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(temperature=0.8, model="gpt-4")
chat_engine = SimpleChatEngine.from_defaults(llm=llm)
```

As you probably won't be using this mode too much in your RAG designs, let's talk about the more advanced options that are available.

### Context mode

**ContextChatEngine** is designed to enhance chat interactions by leveraging our proprietary knowledge. It works by retrieving relevant text from an index based on the user's input, integrating this retrieved information into the system prompt to provide context, and then generating a response with the help of the LLM.

Have a look at *Figure 8.4* for a visual representation of this chat mode:

![Figure 8.4 – ContextChatEngine](images/chapter-08-figure-08-4-contextchatengine.jpg)

*Figure 8.4 – ContextChatEngine*

There are several parameters that we can customize for this chat engine:

- **retriever**: The actual retriever that's used to retrieve relevant text from the index based on the user's message. When the chat engine is initialized directly from the index, it will use the default retriever for that particular index type
- **llm**: An instance of an LLM, which will be used for generating responses
- **memory**: A **ChatMemoryBuffer** object, which is used to store and manage the chat history
- **chat_history**: This is an optional list of **ChatMessage** instances representing the history of the conversation. It can be used to maintain continuity in a conversation. This history includes all messages that have been exchanged in the chat session, including both user and chatbot messages. For instance, it can be used to continue a conversation from a certain point. A **ChatMessage** object contains three attributes:
  - **role**: This defaults to *user*
  - **content**: The actual message
  - Any optional arguments provided via **additional_kwargs**
- **prefix_messages**: A list of **ChatMessage** instances that may be used as predefined messages or prompts before the actual user message. This can be useful for setting a particular tone or context for the chat
- **node_postprocessors**: An optional list of **BaseNodePostprocessor** instances for further processing the nodes retrieved by the retriever. This can be used to implement guardrails, scrub sensitive information from the context, or make any other adjustments to the retrieved nodes if required
- **context_template**: A string template that can be used to format the prompt that feeds the context to the LLM
- **callback_manager**: An optional **CallbackManager** instance for managing callbacks during the chat process. This is useful for tracing and debugging purposes
- **system_prompt**: An optional string that's used as a system prompt, providing initial context or instructions for the chatbot
- **service_context**: An optional **ServiceContext** instance, which can be used to make additional customizations to the chat engine

To implement **ContextChatEngine**, we must load our data and build an index, then optionally configure the chat engine with different parameters as needed.

Here's a quick example based on our sample data files, which can be found in the **ch8/files** subfolder in this book's GitHub repository:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)

chat_engine = index.as_chat_engine(chat_mode="context")
chat_engine.chat_repl()
```

### Condense question mode

**CondenseQuestionChatEngine** addresses a common challenge in conversational AI: maintaining context across multiple turns of conversation. In many chat scenarios, users refer to previous parts of the conversation using pronouns or incomplete references. For example, a user might first ask, "What is machine learning?" and then follow up with, "How is it used in healthcare?"

The challenge here is that the second question, "How is it used in healthcare?", lacks context when considered in isolation. The pronoun "it" refers to "machine learning" from the previous question, but without that context, the question becomes ambiguous.

**CondenseQuestionChatEngine** solves this by using an LLM to reformulate the current question based on the chat history, creating a standalone question that incorporates the necessary context. In our example, the condensed question might become, "How is machine learning used in healthcare?"

*Figure 8.5* illustrates how this chat mode works:

![Figure 8.5 – CondenseQuestionChatEngine](images/chapter-08-figure-08-5-condensequestionchatengine.jpg)

*Figure 8.5 – CondenseQuestionChatEngine*

This approach ensures that each query to the retrieval system is self-contained and contextually complete, leading to more accurate and relevant results.

Here's how to implement this chat mode:

```python
chat_engine = index.as_chat_engine(chat_mode="condense_question")
chat_engine.chat_repl()
```

### Condense plus context mode

**CondensePlusContextChatEngine** combines the benefits of both the condense question and context modes. It first condenses the user's question based on chat history (like **CondenseQuestionChatEngine**), then retrieves relevant context and includes it in the response generation (like **ContextChatEngine**).

*Figure 8.6* shows how this mode operates:

![Figure 8.6 – CondensePlusContextChatEngine](images/chapter-08-figure-08-6-condensepluscontextchatengine.jpg)

*Figure 8.6 – CondensePlusContextChatEngine*

This mode provides the most comprehensive approach to conversational RAG, ensuring both contextual continuity and access to relevant information from your knowledge base.

```python
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
chat_engine.chat_repl()
```

## Implementing agentic strategies in our apps

While chat engines provide excellent conversational capabilities, **agents** take this a step further by adding the ability to use tools and make decisions about which actions to take. Agents can reason about problems, plan solutions, and execute complex workflows using multiple tools.

### Understanding agents

An **agent** is an autonomous system that can:
- Understand user requests
- Plan a sequence of actions
- Use tools to gather information or perform tasks
- Reason about the results
- Provide comprehensive responses

LlamaIndex provides several agent implementations, each with different capabilities and use cases.

### OpenAI Agent

**OpenAIAgent** leverages OpenAI's function calling capabilities to create powerful, tool-using agents. It can automatically decide which tools to use based on the user's query and can chain multiple tool calls together.

```python
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Create tools
vector_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching through documents using semantic similarity"
    )
)

summary_tool = QueryEngineTool(
    query_engine=summary_index.as_query_engine(),
    metadata=ToolMetadata(
        name="summary_search",
        description="Useful for getting comprehensive summaries of documents"
    )
)

# Create agent
agent = OpenAIAgent.from_tools(
    tools=[vector_tool, summary_tool],
    verbose=True
)

# Use the agent
response = agent.chat("Can you search for information about machine learning and then summarize the key points?")
```

### ReAct Agent

**ReActAgent** implements the ReAct (Reasoning and Acting) paradigm, where the agent alternates between reasoning about the problem and taking actions. This approach makes the agent's decision-making process more transparent and interpretable.

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools=[vector_tool, summary_tool],
    verbose=True
)

response = agent.chat("What are the main applications of artificial intelligence?")
```

### Custom Tools

You can create custom tools for your agents to extend their capabilities:

```python
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

# Create tools from functions
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

# Create agent with custom tools
agent = OpenAIAgent.from_tools(
    tools=[multiply_tool, add_tool, vector_tool],
    verbose=True
)

response = agent.chat("What is 25 multiplied by 4, and then add 10 to the result?")
```

### Database Tools

For applications that need to query databases, LlamaIndex provides database tools:

```python
from llama_index.tools.database import DatabaseToolSpec
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine

# Create database connection
engine = create_engine("sqlite:///example.db")
sql_database = SQLDatabase(engine)

# Create database tool
database_tool_spec = DatabaseToolSpec(sql_database=sql_database)
database_tools = database_tool_spec.to_tool_list()

# Create agent with database tools
agent = OpenAIAgent.from_tools(
    tools=database_tools + [vector_tool],
    verbose=True
)

response = agent.chat("What are the top 5 customers by revenue in our database?")

### Advanced Agent Strategies

#### LLM Compiler Agent

The **LLM Compiler Agent** is an advanced agent that can plan and execute multiple tool calls in parallel, significantly improving performance for complex queries that require multiple operations.

```python
from llama_index.packs.agents_llm_compiler import LLMCompilerAgentPack

# Create the LLM Compiler agent pack
agent_pack = LLMCompilerAgentPack(
    tools=[vector_tool, summary_tool, database_tools[0]],
    llm=llm
)

# Use the agent
response = agent_pack.run("Search for information about AI, summarize it, and check our database for related projects")
```

#### Multi-Agent Systems

You can create systems with multiple specialized agents:

```python
# Create specialized agents
research_agent = OpenAIAgent.from_tools(
    tools=[vector_tool],
    system_prompt="You are a research assistant specialized in finding and analyzing information."
)

data_agent = OpenAIAgent.from_tools(
    tools=database_tools,
    system_prompt="You are a data analyst specialized in querying and analyzing database information."
)

# Coordinate between agents
def multi_agent_workflow(query):
    # First, get research information
    research_result = research_agent.chat(f"Research information about: {query}")

    # Then, get relevant data
    data_result = data_agent.chat(f"Find data related to: {query}")

    # Combine results
    final_prompt = f"""
    Based on the research: {research_result}
    And the data: {data_result}
    Provide a comprehensive analysis of: {query}
    """

    return research_agent.chat(final_prompt)

result = multi_agent_workflow("machine learning trends")
```

## Hands-on – implementing conversation tracking for PITS

Now let's implement conversation tracking for our PITS (Personal Intelligent Tutoring System) project. This will allow users to have ongoing conversations with their AI tutor while maintaining context across sessions.

Let's create the **conversation_manager.py** module:

```python
import os
import json
import streamlit as st
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from global_settings import INDEX_STORAGE, CONVERSATION_FILE
```

You'll notice that in the first part of the code, we imported a lot of components. The **os** and **json** modules will be used for the chat persistence feature. The specific LlamaIndex elements will be used to implement the agent with all its required components.

We also imported the **INDEX_STORAGE** and **CONVERSATION_FILE** locations from the **global_settings.py** module. Because the chat conversation will be implemented using Streamlit, we also have to import the **streamlit** library.

Next, let's have a look at the **load_chat_store** function, which is responsible for resuming the previous conversation by loading the chat history from the local storage file specified by **CONVERSATION_FILE**:

```python
def load_chat_store():
    try:
        chat_store = SimpleChatStore.from_persist_path(
            CONVERSATION_FILE
        )
    except FileNotFoundError:
        chat_store = SimpleChatStore()
    return chat_store
```

As we can see, the **load_chat_store** function tries to retrieve the conversation history from the local storage file. If the storage file does not exist, a new empty **chat_store** is created. The function returns **chat_store**.

The next function is responsible for displaying the entire conversation history in the Streamlit interface:

```python
def display_messages(chat_store, container):
    with container:
        for message in chat_store.get_messages(key="0"):
            with st.chat_message(message.role):
                st.markdown(message.content)
```

The **display_messages** function takes a chat store and a Streamlit container as arguments. It extracts all messages from the chat store using **get_messages()**. The function iterates over and displays each message from the chat store, assigning appropriate roles – *user* or *assistant* – to each.

The messages are displayed in the Streamlit container using Streamlit's **chat_message()** method, which has the advantage of automatically adding a corresponding icon for each role.

The next function is responsible for initializing the agent. This function takes five arguments:

- **user_name**: The name of the user – to enable a more personal experience.
- **study_subject**: The topic covered by the study materials.
- **chat_store**: Used to initialize the conversation history.
- **container**: This is the Streamlit container where the chat conversation will be displayed. It's not used by this function itself and instead passed further to the **display_messages** function.
- **context**: This is the content of the current slide being displayed in the training interface. This context will be fed into the agent's system prompt to ground any answer on the current context of the user.

Let's see the first part of the function:

```python
def initialize_chatbot(user_name, study_subject, chat_store, container, context):
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="0"
    )
```

Here, we have defined a **ChatMemoryBuffer** object for the agent, specifying the **chat_store** attribute containing the conversation history. We used the same **chat_store_key** as before. This is important to allow the agent to correctly retrieve the chat history.

Next, we'll prepare the tools for the agent:

```python
    storage_context = StorageContext.from_defaults(
        persist_dir=INDEX_STORAGE
    )
    index = load_index_from_storage(
        storage_context,
        index_id="vector"
    )

    study_materials_engine = index.as_query_engine(
        similarity_top_k=3
    )

    study_materials_tool = QueryEngineTool(
        query_engine=study_materials_engine,
        metadata=ToolMetadata(
            name="study_materials",
            description=(
                f"Provides official information about "
                f"{study_subject}. Use a detailed plain "
                f"text question as input to the tool."
            ),
        )
    )
```

Here, we first retrieved our vector index by using a **StorageContext** object and the **load_index_from_storage()** method. We had to specify the *ID* of the index – *vector* – because in our case, the storage contains more than one index.

After loading the index, we created a simple query engine configured with **similarity_top_k=3** and then created a **QueryEngineTool** utility, providing a proper description in its metadata so that the agent can *understand* its purpose and usage. The top-k similarity parameter is set to **"3"** to retrieve the three most relevant pieces of information from the index.

The next part will initialize **OpenAIAgent**:

```python
    agent = OpenAIAgent.from_tools(
        tools=[study_materials_tool],
        memory=memory,
        system_prompt=(
            f"Your name is PITS, a personal tutor. Your "
            f"purpose is to help {user_name} study and "
            f"better understand the topic of: "
            f"{study_subject}. We are now discussing the "
            f"slide with the following content: {context}"
        )
    )

    display_messages(chat_store, container)
    return agent
```

In the preceding code, we initialized **OpenAIAgent** while providing **QueryEngineTool**, **memory**, and **system_prompt** as arguments. This prompt is used to provide the LLM with background information to contextualize its responses, ensuring they are relevant to the current discussion topic and the user's study needs.

As you can see, I've tried to keep the code as simple as possible. Many things could be improved in this implementation. After initializing the agent, we call **display_messages** to display the existing conversation.

Our last function is responsible for handling the actual conversation. It takes three arguments:

- **agent**: The agent engine that will be used to run the chat
- **chat_store**: The **chat_store** argument that will be used to persist the conversation
- **container**: The Streamlit container where the messages will be displayed

Let's have a look at the code:

```python
def chat_interface(agent, chat_store, container):
    prompt = st.chat_input("Type your question here:")

    if prompt:
        with container:
            with st.chat_message("user"):
                st.markdown(prompt)

            response = str(agent.chat(prompt))

            with st.chat_message("assistant"):
                st.markdown(response)

        chat_store.persist(CONVERSATION_FILE)
```

This **chat_interface** function displays a chat input widget using Streamlit's **chat_input()** method. Upon receiving input, it does the following:

- Adds the user's question to the chat interface in the specified container
- Calls the chat method of **OpenAIAgent** to process the question and generate a response
- Adds the chatbot's response to the chat interface in the specified container
- Persists the new conversation to **CONVERSATION_FILE** using the chat store's persist method to ensure continuity across sessions

That's it for now. We'll talk about more of the features of PITS in the next few chapters.

## Summary

This chapter provided an in-depth exploration of building chatbots and agents with LlamaIndex. We covered **ChatEngine** for conversation tracking and different built-in chat modes, such as simple, context, condense question, and condense plus context.

Then, we explored different agent architectures and strategies using **OpenAIAgent**, **ReActAgent**, and the more advanced LLMCompiler agent. Key concepts such as tools, tool orchestration, reasoning loops, and parallel execution were explained.

We concluded this chapter with a hands-on implementation of conversation tracking for the PITS tutoring application.

Overall, you should now have a comprehensive understanding of leveraging LlamaIndex capabilities to create useful and engaging conversational interfaces.

Throughout the next chapter, we'll discover how to customize our RAG pipeline and provide a straightforward guide to deploying it with Streamlit. We'll also explore advanced tracing methods for seamless debugging and unravel strategies for evaluating our applications.
```
