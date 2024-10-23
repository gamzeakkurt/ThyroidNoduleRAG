# Thyroid Nodule Insights A RAG System Using Llama2 via Hugging-Face

```markdown
# Thyroid Nodules Q&A System

This project implements a Q&A system that utilizes the Llama-2 model from Hugging Face along with retrieval-augmented generation (RAG) techniques. The system is designed to provide accurate information about thyroid nodules based on indexed documents.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [License](#license)

## Installation

Make sure you have the following dependencies installed:

```bash
pip install llama-index huggingface_hub transformers langchain torch
```

## Getting Started

### Hugging Face Login

Before starting the project, create a Hugging Face user token to access the Llama and embedding models. This token is unique to your account, so do not share it with anyone. Once you've generated your token, enter it in the "Token ID" field and uncomment the code below:

```python
# login(token="YOUR_TOKEN_ID")
!huggingface-cli login
```

### Reading Documents

The code reads and loads all the documents stored in the folder "Thyroid Nodules Documents" into memory. These documents will be used by the RAG system to retrieve relevant information during queries.

```python
documents = SimpleDirectoryReader("Thyroid Nodules Documents").load_data()
```

## Code Overview

### Importing Libraries

The following libraries are imported to facilitate the creation of vector indices, document reading, and integration with Hugging Face models:

```python
# Import classes for creating vector indices and reading documents
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
# Import Hugging Face LLM for text generation and embeddings
from llama_index.llms.huggingface import HuggingFaceLLM
# Import class for creating simple input prompts for LLM queries
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Import PyTorch for tensor operations and deep learning
import torch
# Import ServiceContext for managing configurations and services in llama_index
from llama_index.core import ServiceContext
# Import embeddings class from LangChain for using Hugging Face models
from langchain_community.embeddings import HuggingFaceEmbeddings
# Import LangchainEmbedding for generating embeddings with LangChain models
from llama_index.embeddings.langchain import LangchainEmbedding
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
```

### Configuring the Llama-2 Model

The code initializes the Llama-2-7b-chat model along with its tokenizer, loaded with 16-bit precision and configured for quantization to reduce memory usage.

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    torch_dtype=torch.float16,
)

# Pass the model and tokenizer to HuggingFaceLLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer
)
```

### Initializing the Embedding Model

A sentence embedding model is initialized using the `all-mpnet-base-v2` variant from the Sentence Transformers library.

```python
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

### Configuring Settings

The initialized language model and embedding model are prepared for subsequent processing:

```python
Settings.llm = llm
Settings.embed_model = LangchainEmbedding(embed_model)
```

### Creating a Vector Store Index

The `VectorStoreIndex` is created from the loaded documents, enabling efficient retrieval:

```python
index = VectorStoreIndex.from_documents(documents, service_context=Settings)
```

### Querying the Index

Finally, you can query the index for information on thyroid nodules:

```python
query_engine = index.as_query_engine()
response = query_engine.query("What are the Thyroid Nodules?")
print(response)
```

## Usage

You can ask any questions about thyroid nodules. The system retrieves relevant information from the indexed documents based on your queries.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Explanation:
- **Structure**: The README is structured with headings and subheadings for easy navigation.
- **Installation**: A section for installation instructions is included.
- **Getting Started**: Details on logging into Hugging Face and reading documents are provided.
- **Code Overview**: Key sections of the code are summarized for clarity.
- **Usage**: Briefly explains how users can interact with the system.

Feel free to adjust any sections or add more details as needed!
