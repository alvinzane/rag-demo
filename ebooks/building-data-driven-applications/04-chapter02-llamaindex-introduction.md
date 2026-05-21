# Chapter 2: LlamaIndex: The Hidden Jewel - An Introduction to the LlamaIndex Ecosystem

Now that you've got a solid understanding of what **large language models** (**LLMs**) are and what they can (and cannot) do. It's time to discover how **LlamaIndex** can take your interactive AI applications to the next level. We'll explore how **retrieval-augmented generation** (**RAG**) using LlamaIndex can provide the missing link between the vast knowledge of LLMs and your proprietary data.

In this chapter, we will cover the following main topics:

- Optimizing language models – The symbiosis of fine-tuning, RAG, and LlamaIndex
- Discovering the advantages of progressively disclosing complexity
- Introducing **personalized intelligent tutoring system** (**PITS**) – our hands-on LlamaIndex project
- Preparing our coding environment
- Familiarizing ourselves with the structure of the LlamaIndex code repository

## Technical requirements

The following elements will be required for this chapter:

- *Python 3.11* (https://www.python.org/)
- *Git* (https://git-scm.com/)
- *LlamaIndex* (https://github.com/run-llama/llama_index)
- *OpenAI account* and an *API key*
- *Streamlit* (https://github.com/streamlit/streamlit)
- *PyPDF* (https://pypi.org/project/pypdf/)
- *DOC2Txt* (https://github.com/ankushshah89/python-docx2txt/blob/master/docx2txt/docx2txt.py)

All the sample code snippets presented throughout this book as well as the entire project code base can be found in this GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

## Optimizing language models – the symbiosis of fine-tuning, RAG, and LlamaIndex

In the previous chapter, we saw that vanilla LLMs have some limitations right outside of the box. Their knowledge is static and they occasionally spit out nonsense. We also learned about RAG as a potential way to mitigate these issues. Blending **prompt engineering** techniques with programmatic methods, RAG can elegantly solve many of the LLM shortcomings.

**What is prompt engineering?**

Prompt engineering involves crafting text inputs designed to be effectively processed by a **generative AI** (**GenAI**) model. Composed in natural language, these prompts describe the specific tasks to be carried out by the AI. We'll have a much deeper conversation on this topic during *Chapter 10*, *Prompt Engineering Guidelines and Best Practices*.

### Is RAG the only possible solution?

Of course not. Another approach is to fine-tune the AI model, which involves additional training on proprietary data to adapt the LLM and embed new data. It takes a model that is pre-trained on a general collection of data and continues its training on a more specialized dataset. This specialized dataset can be tailored to a particular domain, language, or set of tasks that you are interested in. The result is a model that maintains its broad knowledge base while gaining expertise in a specific area.

Take a look at Figure 2.1 for a graphical explanation of the process.

![Figure 2.1 – An illustration of the LLM fine-tuning process](images/chapter-02-figure-02-1-llm-fine-tuning-process.jpg)

*Figure 2.1 – An illustration of the LLM fine-tuning process*

Fine-tuning can improve performance but has drawbacks, such as being expensive, requiring large datasets, and being difficult to update with fresh information. It also has the disadvantage of permanently altering the original AI model, which makes it inappropriate for personalizing purposes. Think of the original AI model as a classic recipe for a beloved dish. Fine-tuning this model is akin to modifying the traditional recipe to suit specific tastes or requirements. While these changes can make the dish more suitable for some, they also fundamentally alter the original recipe.

> **Note**
> 
> Not all fine-tuning methods permanently alter the base AI model. Take **Low-Rank Adaptation** (**LoRA**) for example. LoRA is a fine-tuning method for LLMs that offers a more efficient approach compared to traditional **full fine-tuning**. In full fine-tuning, all layers of a neural network are optimized, which, while effective, is resource-intensive and time-consuming. LoRA, on the other hand, involves fine-tuning only two smaller matrices that approximate the larger weight matrix of the pre-trained LLM. In the LoRA method, the original weights of the model are *frozen*, meaning they are not directly updated during the fine-tuning process. The changes to the model's behavior are achieved by the addition of these low-rank matrices. This approach allows for the original model to be preserved, while still enabling it to be adapted for new tasks or improved performance. You can find more information on this method here: https://ar5iv.labs.arxiv.org/html/2106.09685.

Even though LoRA is more efficient in terms of memory usage compared to full fine-tuning, it still requires computational resources and expertise to implement and optimize effectively, which might be a barrier for some users. Using fine-tuning to create a more personalized experience for a large number of different users requires re-running the tuning process for every user, which is definitely not cost-effective.

I'm not trying to say that RAG is a better alternative to LLM fine-tuning. In fact, RAG and fine-tuning are complementary techniques that are often used together. However, to rapidly incorporating changing data and personalization, RAG is preferable.

### What LlamaIndex does

With LlamaIndex, you can rapidly create *smart* LLMs that can adapt to your specific use case. Instead of relying only on their generic pre-trained knowledge, you can inject targeted information so that they give you accurate, relevant answers. It provides an easy way to connect external datasets to LLMs such as GPT-4, Claude, and Llama. LlamaIndex builds a bridge between your custom knowledge and the vast capabilities of LLMs.

> **Note**
> 
> Created in 2022 by Princeton University graduate and entrepreneur Jerry Liu, the *LlamaIndex framework* has quickly become very popular in the developer community. LlamaIndex allows you to take advantage of the computational power and language understanding capabilities of LLMs while focusing their responses on specific, reliable data. This unique combination enables businesses and individuals to get the most out of their AI investments, as they can use the same underlying technology for a wide array of specialized applications.

For example, you could index a collection of your company's documents. Then, when you ask questions related to your business, the LLM augmented with LlamaIndex provides responses based on real data rather than just making up vague answers!

The result is that you get all the expressive power of LLMs while greatly reducing the amount of incorrect or irrelevant information. LlamaIndex guides the LLM to pull from trusted sources you provide, and these sources could contain both *structured* and *unstructured* data. In fact, as we will see in the next chapters, the framework can ingest data from pretty much *any* data source available. That's pretty cool, right?

If you are not already thinking about the many possible uses for this framework, let me give you some quick ideas. With LlamaIndex, you could do the following:

- **Build a search engine for your document collection**: One of its most powerful applications is the ability to index all your documents – they could be PDFs, Word files, Notion documents, GitHub repos, or other formats. Once indexed, you can query the LLM to search for specific information, making it a powerful search engine tailored specifically for your resources
- **Create a company chatbot with customized knowledge**: If your business has specific jargon, policies, or expertise, you can make the LLM *understand* these nuances. The chatbot could then handle a range of queries, from basic customer service questions to more specialized interactions that would typically require human expertise
- **Generate summaries of large reports or papers**: If your organization deals with lengthy documents or reports, LlamaIndex can be used to feed the LLM with their contents. Then, you can ask the LLM to generate concise summaries, capturing the most important points
- **Develop a smart assistant for complex workflows**: By training the LLM on the nuances of multi-step tasks or procedures unique to your organization, you can transform it into a smart assistant data agent that provides valuable insights and guidance

And these are just the tip of the iceberg.

In addition, Figure 2.2 shows how implementing smart RAG strategies can offset some of the costs associated with fine-tuning the model on a specific domain.

![Figure 2.2 – The relative costs of updating data in a pre-trained LLM](images/chapter-02-figure-02-2-relative-costs-updating-data.jpg)

*Figure 2.2 – The relative costs of updating data in a pre-trained LLM*

Before we dive deeper into the applications and use cases of the LlamaIndex framework, let's talk a bit about the architecture and the design principles behind it!

## Discovering the advantages of progressively disclosing complexity

The creator of LlamaIndex wanted to make it accessible to everyone – from beginners just getting started with LLMs all the way to expert developers building complex systems. That's why LlamaIndex uses a design principle called **progressive disclosure of complexity**. Don't worry about the fancy name – it just means that the framework starts simple and gradually reveals more advanced features when you need them.

When you first use LlamaIndex, it feels like magic! With just a few lines of code, you can connect data and start querying the LLM. Under the hood, LlamaIndex converts the data into an efficient index that the LLM can use.

Have a look at this very simple example that first loads a set of text documents from a local directory. It then builds an index over the documents and queries that index to get a summarized view of the documents based on a natural language query:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('files').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query(
    "summarize each document in a few sentences"
)
print(response)
```

It's that simple. Just six lines of code!

> **Note**
>
> Don't try to run the code just yet. It's more for illustration purposes. There is a bit of environmental preparation we need to handle before that. Don't worry, we'll cover that a bit later in this chapter and then you'll be ready to go.

As you use LlamaIndex more, you will uncover its more powerful capabilities. There are plenty of parameters you can tweak. You can select specialized index structures optimized for different uses, carry out detailed cost analyses for different prompt strategies, customize query algorithms, and much more.

But LlamaIndex always starts you off gently before getting into more detailed workings, and for quick and simple projects, you don't need to go much deeper than that. This way, both beginners and experts can benefit from its versatility and capabilities.

Now, let's go on a quick tour of our hands-on project and then start prepping for the fun part: writing the code.

### An important aspect to consider

As you go further through this book, and you will most likely want to experiment based on the examples it gives, you need to keep one very important point in mind. By default, the LlamaIndex framework is configured to use AI models provided by OpenAI. Although these models are extremely powerful and versatile, they incur costs. Many of the LlamaIndex functionalities presented in this book, be it metadata extraction, indexing, retrieval, or response synthesis, are based on either LLMs or embedding models. I have tried to use as simple examples as possible with small sample datasets in an attempt to limit these costs as much as possible.

> **Note**
>
> I strongly advise you to keep a close eye on the OpenAI API consumption. In case you don't already have it, the link where you can monitor the API usage is here: https://platform.openai.com/usage. I also advise you to be careful from a privacy perspective. These issues are discussed in more detail in *Chapters 4* and *5*.

Alternatively, if you want to avoid both the costs of using an external LLM and the potential privacy risks, you can apply the methods described in *Chapter 9*, *Customizing and Deploying Our LlamaIndex Project*. It is important to note, however, that all examples provided in the book are written and tested using the default models provided by OpenAI. There is a (quite likely) possibility that some examples may not work as well – or at all – running on locally hosted alternatives.

## Introducing PITS – our LlamaIndex hands-on project

*Nothing beats learning by doing.*

So, I've cooked up a fun and useful project for us to start using LlamaIndex!

Here, we will introduce PITS. Wouldn't it be cool to have an AI tutor that helps you learn new concepts interactively? Well, we're going to build one together!

### Here's how it will work

First, you will introduce yourself to PITS. You'll have the chance to describe the topic you want to learn about and specify any personal learning preferences you may have.

Then, you will be able to upload any existing study materials you may have on the topic. PITS will accept and ingest any PDFs, Word documents, or text files you may provide.

Based on the documents provided, the tutor will first build a quiz. You'll have the option to complete the quiz. That way, the tutor will be able to gauge your current knowledge of the topic and adjust the learning experience.

Our nifty tutor will then build learning material for you. This will consist of slides and narration for each slide. The training material will be divided into chapters.

Then, your learning journey begins. During each learning session, PITS you will advance through the chapters, presenting each topic in your preferred style and adapting to your knowledge level.

After each concept is explained, you'll have a chance to ask for more explanations or examples to learn more about the topic. It will answer your questions, create quizzes, explain concepts, and adapt responses based on your needs.

The best part is that your entire conversation with the agent will be recorded. It will remember both your questions and its own answers so it won't repeat itself or lose the conversation context.

Too tired to continue in one session? Not a problem. When you're ready to start another lesson, it will just resume from where you left off and give you a summary of the previous discussion.

But, hey! They say a picture's worth a thousand words, right?

You'll find an overview in Figure 2.3.

![Figure 2.3 – An overview of the PITS workflow](images/chapter-02-figure-02-3-pits-workflow-overview.jpg)

*Figure 2.3 – An overview of the PITS workflow*

It doesn't really get more customized than this. This is the ultimate learning experience.

As you can imagine, PITS needs to be smart on several fronts. It needs to be able to do the following:

- Understand and index the study materials we provide
- Converse fluently with users and retain the context
- Teach effectively based on the indexed knowledge

LlamaIndex will help with the first part by ingesting the study material. The user will be able to upload any relevant training material such as manuals, slides or even student notes, and sample questions.

For the second part, we'll mostly use the capabilities of GPT-4 to power the actual teaching interactions.

However, the foundation will be the knowledge augmentation capabilities of LlamaIndex. Pretty neat, right? We'll have a personally customized tutor!

> **Note**
>
> I'm not sure whether you've read my biography, but I work as a trainer. The moment I first learned of the power of GenAI and discovered GPT-3, I knew exactly that a few years from now, systems such as PITS would emerge sooner or later. I was thrilled about their potential to provide free, quality education to people around the world, regardless of their location, background, or financial status. Later, when I discovered RAG and tools such as LlamaIndex, I became convinced that they would appear rather sooner than later.

Okay, enough daydreaming – let's start setting up the pieces.

## Preparing our coding environment

Before we embark on the LlamaIndex coding journey, it's essential to set up our development environment properly. This setup is the first step toward ensuring that we can smoothly run through the examples and exercises I've prepared for you.

> **Note**
>
> To maintain simplicity and ensure consistency across all examples, I've designed the sample code to be run in a local Python environment. I'm aware that many of you are fond of using web-based coding environments such as Google Colab and Jupyter Notebooks for your coding projects, so I kindly ask for your understanding if these examples do not directly translate to or run in these platforms. My goal was to keep our setup straightforward, allowing us to focus on the learning experience without compatibility concerns. Thank you for your understanding and happy coding!

Let's quickly get our computer set up for some cool LlamaIndex coding.

### Installing Python

You'll need a Python 3.7+ environment. I recommend Python 3.11 if possible.

If you don't have Python, install it from https://www.python.org. If you already have an older version, you can upgrade or install a newer Python version side by side.

For a coding environment, my personal preference is **NotePad++** (https://notepad-plus-plus.org/), which is not quite an IDE but is very fast. However, you can also use Microsoft's **VSCode** (https://code.visualstudio.com/), **PyCharm** (https://www.jetbrains.com/pycharm/), or anything else you prefer.

### Installing Git

Before we proceed, it's important to have Git installed. Git is a version control system that lets you manage changes to your code and collaborate with others. It's also essential for cloning code repositories, like the one we'll be using in this book.

Head over to the official Git website (https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and download the installer for your operating system.

Follow the installation steps, and you should have Git up and running in no time.

All the sample code snippets presented throughout the book as well as the entire project code base can be found in this GitHub repository: https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex.

So, if you want to download the project files locally, once you have finished installing Git, you can simply follow these steps:

1. **Navigate to the desired directory**: Open a new command prompt or terminal window. Use the **cd** command to navigate to the directory where you'd like to store the project. Here is an example:

```bash
cd path/to/your/directory
```

2. **Clone the repository**: Run the following command to clone the GitHub repository:

```bash
git clone https://github.com/PacktPublishing/Building-Data-Driven-Applications-with-LlamaIndex
```

This will download a copy of the project to your local machine.

3. **Enter the project directory**: Navigate into the newly created project folder:

```bash
cd Building-Data-Driven-Applications-with-LlamaIndex
```

As we move forward with our project, you have two options:

- You can either write the code on your own and then compare it with what's in the repository
- Or you can directly explore the code files in the repository to get a better understanding of the code structure

If you correctly performed all of the preceding steps, listing the contents of the current folder should return several subfolders called **chX** – where **X** is the chapter number, and a separate subfolder called **PITS_APP**. The chapter folders contain all sample source files corresponding to each chapter. The **PITS_APP** folder contains the source code for our main project.

### Installing LlamaIndex

Next, let's get the LlamaIndex library installed. At your command prompt, run the following:

```bash
pip install llama-index
```

This will include a LlamaIndex package that contains the core LlamaIndex components as well as a selection of useful integrations. For the most efficient deployment possible, there is also the option of installing just the minimum core components and only the necessary integrations, but for the purpose of this book, the presented option will do just fine.

> **Note**
>
> In case you're already running a version older than v0.10, it is recommended that you start with a fresh install in a virtual environment to avoid any conflicts with the legacy version. You can find detailed instructions here: https://pretty-sodium-5e0.notion.site/v0-10-0-Migration-Guide-6ede431dcb8841b09ea171e7f133bd77.

We're now ready to import and start using it.

### Signing up for an OpenAI API key

Since we'll be using OpenAI's GPT models via LlamaIndex, you'll need an API key to authenticate. Head to https://platform.openai.com and sign up. Once logged in, you can create a new secret API key. Make sure to keep it safe!

LlamaIndex will use this key every time it interacts with OpenAI's models. Because it has to be kept secret, it's a good idea to store it in an environment variable on your local machine.

#### A short guide for Windows users

On Windows, you can accomplish that by following these steps:

1. **Open Environment Variables**: Open the Start menu and search for Environment Variables or right-click on This PC or My Computer and select Properties.

2. Then, click on Advanced system settings followed by the Environment Variables button in the Advanced tab as shown in Figure 2.4:

![Figure 2.4 – Editing Windows environment variables](images/chapter-02-figure-02-4-editing-windows-environment-variables.jpg)

*Figure 2.4 – Editing Windows environment variables*

3. **Create a new environment variable**: In the Environment Variables window, under the User variables section, click the New button.

4. **Enter the variable details**: For the Variable name, enter OPENAI_API_KEY. For Variable value, paste the secret API key you received from OpenAI. See Figure 2.5 for an illustration.

![Figure 2.5 – Creating the OPENAI_API_KEY environment variable](images/chapter-02-figure-02-5-creating-openai-api-key-environment-variable.jpg)

*Figure 2.5 – Creating the OPENAI_API_KEY environment variable*

5. **Confirm and apply**: Click OK to close all of the dialog boxes. You will need to restart your computer for the changes to take effect.

6. **Verify the environment variable**: To ensure the variable is set correctly, open a new command prompt, and run the following:

```bash
echo %OPENAI_API_KEY%
```

This should display the API key you just stored.

#### A short guide for Linux/Mac users

On Linux/Mac, you can accomplish Signing up for an OpenAI API key by following these steps:

1. Run the following command in your terminal, replacing <yourkey> with your API key:

```bash
echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
```

2. Update the shell with the new variable:

```bash
source ~/.zshrc
```

3. Make sure that you have set your environment variable with the following command:

```bash
echo $OPENAI_API_KEY
```

Your OpenAI API key is now securely stored in an environment variable and can be easily accessed by LlamaIndex when needed, without exposing it in your code or system.

> **Note**
>
> While OpenAI provides a free trial option for their GPT models through their API, you'll only receive a limited number of free credits. Currently, the free credit is limited to $5 and expires after 3 months. That should be more than enough to experiment for the purpose of our project and for reading the book. However, If you wish to get serious about building LLM-based applications, you'll have to sign up for a paid account on their platform. Alternatively, you can always choose to use another AI model for LlamaIndex. We will discuss customizing the AI model in more detail in Chapter 10, Prompt Engineering Guidelines and Best Practices.

OK. The backend is all set up. Let's talk about the rest of the stack.

### Discovering Streamlit – the perfect tool for rapid building and deployment!

Before we can build cool apps such as PITS, we need somewhere to … well, build and run them! That's where Streamlit comes in. Streamlit is an awesome open-source Python library that makes it super easy to create and deploy web apps and dashboards.

With just a few lines of Python code, you can build complete web interfaces and see the results instantly. The best part is that Streamlit apps can be deployed nearly anywhere – on servers, on platforms such as Heroku, or even directly from GitHub!

I love Streamlit because it lets me focus on the fun stuff – such as creating PITS with LlamaIndex – rather than fussing over complex web development. For AI experimentation, it's perfect!

We'll primarily use it to create the interface for uploading study guides and interacting with our PITS tutor. For the purpose of the next chapters, we'll be using Streamlit for running and testing our app locally. However, in Chapter 9, Customizing and Deploying Our LlamaIndex Project, we will also discover how we can easily deploy our app using Streamlit Share or any other hosting service you prefer.

Streamlit has tons of cool capabilities such as data frames, charts, and widgets – but don't worry about learning it all now. As we build up features, I'll explain the relevant parts so you can gain Streamlit skills along the way!

### Installing Streamlit

Lastly, we need to install the Streamlit library:

```bash
pip install streamlit
```

Great! We have our backend tool (LlamaIndex), our frontend layer (Streamlit), and our goal (PITS). It's time for a final touch.

### Finishing up

Because our project should be able to ingest PDF and DOCX documents, we will also need to install two additional libraries:

```bash
pip install pypdf
pip install docx2txt
```

That's it! Our environment is LlamaIndex ready.

Let's recap what we have:

- Python 3.11
- Git
- LlamaIndex package
- OpenAI account and an API key
- Streamlit for app building
- PyPDF and DOC2Txt libraries

### One final check

To verify that everything was installed correctly, open a new command prompt or terminal window, and run the following commands:

```bash
python --version
git --version
pip show llama-index
echo %OPENAI_API_KEY%
pip show streamlit
pip show pypdf
pip show docx2txt
```

A simple way to check whether your environment is ready is to try navigating into the **ch2** subfolder of your local **git** folder and run the file called **sample1.py**:

```bash
python sample1.py
```

You should get a nice summary of the two sample documents provided in the **ch2/files** subfolder if everything has been properly installed.

If anything is missing, please go back and retake the necessary steps before proceeding further. Trust me, you'll avoid a lot of pain and frustration further down the line.

We're all set to start ingesting data, constructing indices with LlamaIndex, and building our PITS tutor app! I don't know about you, but I'm *kid-in-a-candy-store* excited to start experimenting.

In the next chapters, we'll get hands-on with our first LlamaIndex program. This is where the real fun begins! We'll explore ingesting data, constructing indexes, executing queries, and more.

I'll explain each concept and line of code in simple terms along the way. In no time, you'll be implementing the basics like a LlamaIndex pro! Once we've got these fundamentals down, we can start expanding the capabilities of our tutor app.

But first, let's clarify the overall code structure of the framework's GitHub repository.

## Familiarizing ourselves with the structure of the LlamaIndex code repository

Because you'll probably spend a lot of time browsing the official code repository of the LlamaIndex framework, it's good to have an overall image of its general structure. You can always consult the repository here: https://github.com/run-llama/llama_index.

Starting with version 0.10, the code has been thoroughly reorganized into a more modular structure. The purpose of this new structure is to improve efficiency, by avoiding loading any unnecessary dependencies, while also improving readability and overall user experience for developers.

Figure 2.6 describes the main components of the code structure:

![Figure 2.6 – The LlamaIndex GitHub repository code structure](images/chapter-02-figure-02-6-llamaindex-github-repository-structure.jpg)

*Figure 2.6 – The LlamaIndex GitHub repository code structure*

The **llama-index-core** folder serves as the foundational package for LlamaIndex, enabling developers to install the essential framework and then selectively add from over 300 integration packages and different Llama-packs to tailor functionality for their specific application needs.

The **llama-index-integrations** folder of LlamaIndex consists of various add-on packages that extend the functionality of the core framework. These allow developers to customize their build with specific elements such as custom LLMs, data loaders, embedding models, and vector store providers to best fit their application's requirements. We'll cover some of these integrations later in our book, starting with *Chapter 4*, *Ingesting Data into Our RAG Workflow*.

The **llama-index-packs** folder contains more than 50 Llama packs. Developed and constantly improved by the LlamaIndex developer community, these packs serve as ready-made templates designed to kickstart a user's application. We'll talk about them in more detail during *Chapter 9*, *Customizing and Deploying Our LlamaIndex Project*.

The **llama-index-cli** folder is used by the LlamaIndex command-line interface, which we will also cover briefly during *Chapter 9*, *Customizing and Deploying Our LlamaIndex Project*.

The last section, called **OTHERS** in Figure 2.6, consists of two folders that currently contain fine-tuning abstractions and some experimental features that we will not cover in this book.

> **Note**
>
> The subfolders in **llama-index-integrations** and **llama-index-packs** represent individual packages. The folder name corresponds to the package name. For example, the **llama-index-integrations/llms/llama-index-llms-mistralai** folder corresponds to the **llama-index-llms-mistralai** PyPI package.

Following this example, there is something you need to do before you import and use the **mistralai** package in your code like this:

```python
from llama_index.llms.mistralai import MistralAI
```

You'll have to first install the corresponding PyPI package by running the following:

```bash
pip install llama-index.llms.mistralai
```

Don't worry too much about missing any necessary packages for the examples included in the book, as you will find them nicely listed at the beginning of each chapter under the *Technical requirements* heading.

## Summary

In this chapter, we introduced LlamaIndex, a framework for connecting LLMs to external datasets. We discovered how LlamaIndex allows LLMs to incorporate real-world knowledge into their responses.

The chapter discussed the benefits of LlamaIndex over fine-tuning, such as easier updating and personalization. It introduced the concept of progressive disclosure of complexity, where LlamaIndex starts simple but reveals advanced capabilities when needed.

The chapter then presented an overview of the hands-on project PITS, a personalized intelligent tutoring system. It covered setting up the required tools such as Python, Git, and Streamlit, and getting an OpenAI API key. The chapter finished by verifying that the environment is ready for building LlamaIndex apps.

We're now ready to continue our journey and proceed with a more technical understanding of the inner workings of the LlamaIndex framework. See you in the next chapter!
