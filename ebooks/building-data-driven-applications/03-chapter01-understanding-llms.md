# Chapter 1: Understanding Large Language Models

If you are reading this book, you have probably explored the realm of **large language models** (**LLMs**) and already recognize their potential applications as well as their pitfalls. This book aims to address the challenges LLMs face and provides a practical guide to building data-driven LLM applications with LlamaIndex, taking developers from foundational concepts to advanced techniques for implementing **retrieval-augmented generation** (**RAG**) to create high-performance interactive **artificial intelligence** (**AI**) systems augmented by external data.

This chapter introduces **generative AI** (**GenAI**) and LLMs. It explains how LLMs generate human-like text after training on massive datasets. We'll also overview LLM capabilities, limitations such as outdated knowledge potential for false information, and lack of reasoning. You'll be introduced to RAG as a potential solution, combining retrieval models using indexed data with generative models to increase fact accuracy, logical reasoning, and context relevance. Overall, you'll gain a basic LLM understanding and learn about RAG as a way to overcome some LLM weaknesses, setting the stage for utilizing LLMs practically.

In this chapter, we will cover the following main topics:

- Introducing GenAI and LLMs
- Understanding the role of LLMs in modern technology
- Exploring challenges with LLMs
- Augmenting LLMs with RAG

## Introducing GenAI and LLMs

Introductions are sometimes boring, but here, it is important for us to set the context and help you familiarize yourself with GenAI and LLMs before we dive deep into LlamaIndex. I will try to be as concise as possible and, if the reader is already familiar with this information, I apologize for the brief digression.

### What is GenAI?

**GenAI** refers to systems that are capable of generating new content such as text, images, audio, or video. Unlike more specialized AI systems that are designed for specific tasks such as image classification or speech recognition, GenAI models can create completely new assets that are often very difficult – if not impossible – to distinguish from human-created content.

These systems use **machine learning** (**ML**) techniques such as **neural networks** (**NNs**) that are trained on vast amounts of data. By learning patterns and structures within the training data, generative models can model the underlying probability distribution of the data and sample from this distribution to generate new examples. In other words, they act as big prediction machines.

We will now discuss LLMs, which are one of the most popular fields in GenAI.

### What is an LLM?

One of the most prominent and rapidly advancing branches of GenAI is **natural language generation** (**NLG**) through **LLMs** (Figure 1.1):

![Figure 1.1 – LLMs are a sub-branch of GenAI](images/chapter-01-figure-01-1-llms-sub-branch-genai.jpg)

*Figure 1.1 – LLMs are a sub-branch of GenAI*

LLMs are NNs that are specifically designed and optimized to understand and generate human language. They are *large* in the sense that they are trained on massive amounts of text containing billions or even trillions of words scraped from the internet and other sources. Larger models show increased performance on benchmarks, better generalization, and new emergent abilities. In contrast with earlier, rule-based generation systems, the main distinguishing feature of an LLM is that it can produce novel, original text that reads naturally.

By learning patterns from many sources, LLMs acquire various language skills found in their training data – from nuanced grammar to topic knowledge and even basic common-sense reasoning. These learned patterns allow LLMs to extend human-written text in contextually relevant ways. As they keep improving, LLMs create new possibilities for automatically generating **natural language** (**NL**) content at scale.

During the training process, LLMs gradually learn probabilistic relationships between words and rules that govern language structure from their huge dataset of training data. Once trained, they are able to generate remarkably human-like text by predicting the probability of the next word in a sequence, based on the previous words. In many cases, the text they generate is so natural that it makes you wonder: aren't we humans just a similar but more sophisticated prediction machine? But that's a topic for another book.

One of the key architectural innovations is the **transformer** (that is the *T* in *GPT*), which uses an **attention mechanism** to learn contextual relationships between words. Attention allows the model to learn long-range dependencies in text. It's like if you're listening carefully in a conversation, you pay **attention** to the context to understand the full meaning. This means they *understand* not just words that are close together but also how words that are far apart in a sentence or paragraph relate to each other.

*Attention* allows the model to selectively focus on relevant parts of the input sequence when making predictions, thus capturing complex patterns and dependencies within the data. This feature makes it possible for particularly large transformer models (with many parameters and trained on massive datasets) to demonstrate surprising new abilities such as in-context learning, where they can perform tasks with just a few examples in their prompt. To learn more about transformers and **Generative Pre-trained Transformer** (**GPT**), you can refer to *Improving Language Understanding with unsupervised learning* – Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever (https://openai.com/research/language-unsupervised).

The best-performing LLMs such as GPT-4, Claude 2.1, and Llama 2 contain trillions of parameters and have been trained on internet-scale datasets using advanced **deep learning** (**DL**) techniques. The resulting model has an extensive vocabulary and a broad knowledge of language structure such as grammar and syntax, and about the world in general. Thanks to their unique traits, LLMs are able to generate text that is coherent, grammatically correct, and semantically relevant. The outputs they produce may not always be completely logical or factually accurate, but they usually read convincingly like being written by a human. But it's not all about size. The quality of data and training algorithms – among others – can also play a huge role in the resulting performance of a particular model.

Many models feature a user interface that allows for response generation through prompts. Additionally, some offer an API for developers to access the model programmatically. This method will be our primary focus in the upcoming chapters of our book.

Next up, we'll talk about how LLMs are making big changes in tech. They're helping not just big companies but everyone. Curious? Let's keep reading.

## Understanding the role of LLMs in modern technology

Oh! What good times we are living in. There has never been a more favorable era for small businesses and entrepreneurs. Given the enormous potential of this technology, it's a real miracle that, instead of ending up strictly under the control of large corporations or governments, it is literally within everyone's reach. Now, it's truly possible for almost anyone – even a non-technical person – to realize their ideas and solve problems that until now seemed impossible to solve without a huge amount of resources.

The disruptive potential that LLMs have – in almost all industries – is enormous.

It's true: there are concerns that this technology could replace us. However, technology's role is to make lives easier, taking over repetitive activities. As before, we'll likely do the same things, only much more efficiently and better with LLMs' help. We will do more with less.

I would dare say that LLMs have become the foundation of NLG technology. They can already power chatbots, search engines, coding assistants, text summarization tools, and other applications that synthesize written text interactively or automatically. And their capabilities keep advancing rapidly with bigger datasets and models.

And then, there are also the **agents**. These automated wonders are capable of perceiving and interpreting *stimuli* from the digital environment – and not just digital – to make decisions and act accordingly. Backed by the power of an LLM, intelligent agents can solve complex problems and fundamentally change the way we interact with technology. We'll cover this topic in more detail throughout *Chapter 8*, *Building Chatbots and Agents with LlamaIndex*.

Despite their relatively short existence, LLMs have already proven to be remarkably versatile and powerful. With the right techniques and prompts, their output can be steered in useful directions at scale. LLMs are driving innovation in numerous fields as their generative powers continue to evolve. Their capabilities keep expanding from nuanced dialog to multimodal intelligence. And, at the moment, the LLM-powered wave of innovation across industries and technologies shows no signs of slowing down.

**The Gartner Hype Cycle model** serves as a strategic guide for technology leaders, helping them evaluate new technologies not just on their merits but also in the context of their organization's specific needs and goals (https://www.gartner.com/en/research/methodologies/gartner-hype-cycle).

Judging by current adoption levels, LLMs are currently well into the **Slope of Enlightenment** stage, ready to take off into the **Plateau of Productivity** – where mainstream adoption really starts to take off (Figure 1.2). Companies are becoming more pragmatic about their application, focusing on specialized use cases where they offer the most value:

![Figure 1.2 – The Gartner Hype Cycle](images/chapter-01-figure-01-2-gartner-hype-cycle.jpg)

*Figure 1.2 – The Gartner Hype Cycle*

But, unlike other more specific technologies, LLMs are rather a new form of infrastructure – a kind of ecosystem where new concepts will be able to manifest and, undoubtedly, revolutionary applications will be born.

This is their true potential, and this is the ideal time to learn how to take advantage of the opportunities they offer.

Before we jump into innovative solutions that could maximize LLMs' capabilities, let's take a step back and look at some challenges and limitations.

## Exploring challenges with LLMs

Not all the news is good, however. It's time to also discuss the *darker* side of LLMs.

These models do have important limitations and some collateral effects too. Here is a list of the most important ones, but please consider it non-exhaustive. There may be others not included here, and the order is arbitrarily chosen:

- **They lack access to real-time data.**
  - LLMs are trained on a static dataset, meaning that the information they have is only as up to date as the data they were trained on, which might not include the latest news, scientific discoveries, or social trends.
  - This limitation can be critical when users seek real-time or recent information, as the LLMs might provide outdated or irrelevant responses. Furthermore, even if they cite data or statistics, these numbers are likely to have changed or evolved, leading to potential misinformation.

> **Note**
>
> While recent features introduced by OpenAI, for example, allow the underlying LLM to integrate with Bing to retrieve fresh context from the internet, that's not an inherent feature of the LLM but rather an augmentation provided by the ChatGPT interface.

- This lack of real-time updating also means that LLMs – by themselves – are not suited for tasks such as live customer service queries that may require real-time access to user data, inventory levels, or system statuses, for example.

- **They have no intrinsic way of distinguishing factual truth from falsehoods.**
  - Without proper monitoring, they can generate convincing misinformation. And trust me – they don't do it on purpose. In very simple terms, LLMs are basically just looking for words that fit together.
  - Check out Figure 1.3 for an example of how one of the previous versions of the GPT-3.5 model would produce false information:

![Figure 1.3 – Screenshot from a GPT 3.5-turbo-instruct playground](images/chapter-01-figure-01-3-gpt-3-5-turbo-instruct-playground.jpg)

*Figure 1.3 – Screenshot from a GPT 3.5-turbo-instruct playground*

- As these models stochastically (randomly) generate text, their outputs are not guaranteed to be completely logical, factual, or harmless. Also, the training data inherently biases the model, and LLMs may generate toxic, incorrect, or nonsensical text without warning. Since this data sometimes includes unsavory elements of online discourse, LLMs risk amplifying harmful biases and toxic content present in their training data.

> **Note**
>
> While this kind of result may be easily achieved in a playground environment, using an older AI model, OpenAI's ChatGPT interface uses newer models and employs additional guardrails, thus making these kinds of responses much less probable.

- **They also cannot maintain context and memory over long documents.**
  - An interaction with a vanilla-flavor, standard LLM can prove to be a charm for simple topics or a quick question-and-answer session. But go beyond the context window limit of the model, and you'll soon experience its limitations as it struggles to maintain coherence and may lose important details from earlier parts of the conversation or document. This can result in fragmented or incomplete responses that may not fully address the complexities of a long-form interaction or in-depth analysis, just like a human suffering from *short-term memory loss*.

> **Note**
>
> Although recently released AI models such as Anthropic's Claude 2.1 and Google's Gemini Pro 1.5 have dramatically raised the bar in terms of context window limit, ingesting an entire book and running inference on such a large context may prove to be prohibitive from a cost perspective.

- **LLMs also exhibit unpredictable failures in reasoning and fact retention.** Take a look at Figure 1.4 for a typical logic reasoning problem that proves to be challenging even for newer models such as GPT-4:

![Figure 1.4 – Screenshot from a GPT-4 playground](images/chapter-01-figure-01-4-gpt-4-playground.jpg)

*Figure 1.4 – Screenshot from a GPT-4 playground*

- In this example, the answer is wrong because the only scenario that fits is if Emily is the one telling the truth. The treasure would then be neither in the attic nor in the basement.
- Their capabilities beyond fluent text generation remain inconsistent and limited. Blindly trusting their output without skepticism invites errors.

- **The complexity of massive LLMs also reduces transparency into their functioning.**
  - The lack of interpretability makes it hard to audit for issues or understand exactly when and why they fail. All you get is the output, but there's no easy way of knowing the actual decision process that led to that output or the documented fact in which that particular output is grounded. As such, LLMs still require careful governance to mitigate risks from biased, false, or dangerous outputs.

- **As with many other things out there, it turns out we cannot really call them sustainable. At least not yet.**
  - Their massive scale makes them expensive to train and environmentally costly due to huge computing requirements. And it's not just the training itself but also their usage. According to some estimates, "the water consumption of ChatGPT has been estimated at 500 milliliters for a session of 20-50 queries" – *AMPLIFY, VOL. 36, NO. 8*: *Arthur D. Little's Greg Smith, Michael Bateman, Remy Gillet, and Eystein Thanisch* (https://www.cutter.com/article/environmental-impact-large-language-models). This is not negligible by any means. Think about the countless failed attempts to get an answer from an LLM, then multiply that with the countless users exercising their prompt engineering skills every minute.

- **And here's some more bad news: as models advance in complexity and training techniques, LLMs are rapidly becoming a huge source of machine-generated text.**
  - So huge, in fact, that according to predictions, it will end up almost entirely replacing human-generated text (*Brown, Tom B. et al. (2020). Language Models are Few-Shot Learners. arXiv:2005.14165 [cs.CL]. https://arxiv.org/abs/2005.14165*).
  - In a way, this means they may become the victims of their own success. As more and more data is generated by AI, it gradually *contaminates* the training of new models, decreasing their capabilities.
  - As in biology, any ecosystem that cannot maintain a healthy diversity in its genetic pool will gradually degrade.

*I saved the good news for last.*

What if I told you there is at least one solution that can partially address almost all these problems?

In many ways, a language model is very similar to an operating system. It provides a foundational layer upon which applications can be built. Just as an operating system manages hardware resources and provides services for computer programs, LLMs manage linguistic resources and provide services for various **NL processing** (**NLP**) tasks. Using prompts to interact with them is much like writing code using an Assembly Language. It's a low-level interaction. But, as you'll soon find out, there are more sophisticated and practical ways of using LLMs to their full potential.

It's time to talk about RAG.

## Augmenting LLMs with RAG

Coined for the first time in a 2020 paper, *Lewis, Patrick et al. (2005). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". arXiv:2005.11401 [cs.CL]* (https://arxiv.org/abs/2005.11401), published by several researchers from Meta, RAG is a technique that combines the powers of retrieval methods and generative models to answer user questions. The idea is to first retrieve relevant information from an indexed data source containing proprietary knowledge and then use that retrieved information to generate a more informed, context-rich response using a generative model (Figure 1.5):

![Figure 1.5 – A RAG model](images/chapter-01-figure-01-5-rag-model.jpg)

*Figure 1.5 – A RAG model*

Let's have a look at what this means in practice:

- **Much better fact retention**: One of the advantages of using RAG is its ability to pull from specific data sources, which can improve fact retention. Instead of relying solely on the generative model's own *knowledge* – which is mostly generic – it refers to external documents to construct its answers, increasing the chances that the information is accurate.

- **Improved reasoning**: The retrieval step allows RAG models to pull in information that is specifically related to the question. In general, this would result in more logical and coherent reasoning. This could help overcome limitations in reasoning that many LLMs face.

- **Context relevance**: Because it pulls information from external sources based on the query, RAG can be more contextually accurate than a standalone generative model, which has to rely only on its training data and might not have the most up-to-date or contextually relevant information. Not only that, but you could also get an actual *quote* from the model regarding the source of the actual knowledge used in the answer.

- **Reduced trust issues**: While not foolproof, the hybrid approach means that RAG could, in principle, be less prone to generating completely false or nonsensical answers. That means an increased probability of receiving a valid output.

- **Validation**: It's often easier to validate the reliability of the retrieved documents in an RAG setup by setting up a mechanism to provide a reference to the original information used for generating a response. This could be a step toward more transparent and trustworthy model behavior.

> **A word of caution**
>
> Even if RAG makes LLMs better and more reliable, it doesn't completely fix the issue of them sometimes giving wrong or confusing answers. There is no silver bullet that will completely eliminate all the issues mentioned previously. It's still a good idea to double-check and evaluate their outputs, and we'll talk about ways of doing that later in the book. Because, as you may already know or you've probably guessed by now, LlamaIndex is one of the many ways of augmenting LLM-based applications using RAG. And a very effective one, I should add.

While some LLM providers have started introducing RAG components into their API, such as OpenAI's **Assistants** feature, using a standalone framework such as LlamaIndex provides many more customization options. It also enables the usage of local models, enabling self-hosted solutions and greatly reducing costs and privacy concerns associated with a hosted model.

## Summary

In this chapter, we covered a quick introduction to GenAI and LLMs. You learned how LLMs such as GPT work and some of their capabilities and limitations. A key takeaway is that while powerful, LLMs have weaknesses – such as the potential for false information and lack of reasoning – that require mitigation techniques. We discussed RAG as one method to overcome some LLM limitations.

These lessons provide useful background on how to approach LLMs practically while being aware of their risks. At the same time, you learned the importance of techniques such as RAG to address LLMs' potential downsides.

With this introductory foundation in place, we are now ready to dive into the next chapter where we will explore the LlamaIndex ecosystem. LlamaIndex offers an effective RAG framework to augment LLMs with indexed data for more accurate, logical outputs. Learning to leverage LlamaIndex tools will be the natural next step to harness the power of LLMs in a proficient way.
