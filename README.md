# Thyroid Nodule Insights A RAG System Using Llama2 via Hugging-Face 🦙

The **Thyroid Nodule Insights** project is a sophisticated question-and-answer platform that utilizes the **Llama-2** model from Hugging Face, augmented by **Retrieval-Augmented Generation (RAG)** techniques. This system is designed to provide precise and informative responses regarding thyroid nodules, leveraging a carefully curated set of indexed documents to ensure that users receive accurate and contextually relevant information.

By incorporating advanced natural language processing (NLP) capabilities, the system facilitates the effective retrieval of pertinent information, thereby supporting healthcare professionals, patients, and researchers in their efforts to understand and manage thyroid nodules more effectively.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Code Overview](#code-overview)
- [Usage](#usage)
- [References](#references)
- [License](#license)



## Installation

To get started with the Thyroid Nodules Q&A System, ensure you have the necessary dependencies installed. You can easily install them using the following command:

```
pip install llama-index huggingface_hub transformers langchain torch
pip install pypdf
pip install sentence_transformers
pip install llama-index
pip install llama-index-embeddings-langchain
```


## Getting Started

### Hugging Face Login

Before starting the project, create a Hugging Face user token to access the Llama and embedding models. This token is unique to your account, so do not share it with anyone. Once you've generated your token, enter it in the "Token ID" field and uncomment the code below:

```python
login(token="YOUR_TOKEN_ID")
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
```
### Reading Documents

The code reads and loads all the documents stored in the folder "Thyroid Nodules Documents" into memory. These documents will be used by the RAG system to retrieve relevant information during queries.

```python
documents = SimpleDirectoryReader("Thyroid Nodules Documents").load_data()
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

## References

This project utilizes the following documents and articles related to thyroid nodules:

1. **Thyroid Nodules**  
   - Authors: Mary Jo Welker, Diane Orlov  
   - Journal: American Family Physician  
   - Volume: 67, Issue: 3, Pages: 559-567, Year: 2003  
   - Link: [Read More](https://www.aafp.org/pubs/afp/issues/2003/0201/p559.html)  

 

2. **Evaluation and Management of the Pediatric Thyroid Nodule**  
   - Authors: Jeremy T. Guille, et al.  
   - Journal: The Oncologist  
   - Volume: 20, Issue: 1, Pages: 19-27, Year: 2015
   - Link:[Read More](https://theoncologist.onlinelibrary.wiley.com/doi/pdfdirect/10.1634/theoncologist.2014-0115)
 
3. **Thyroid Nodules**  
   - Authors: Kenneth D. Burman, Leonard Wartofsky  
   - Journal: New England Journal of Medicine  
   - Volume: 373, Issue: 24, Pages: 2347-2356, Year: 2015
   -  Link:[Read More](https://emrc.mui.ac.ir/sites/emrc/files/Journal_Club/Thyroid%20Nodules.pdf)

  

4. **Investigating the Thyroid Nodule**  
   - Authors: H. M. Mehanna, et al.  
   - Journal: BMJ  
   - Volume: 338, Year: 2009
   - Link:[Read More](https://d1wqtxts1xzle7.cloudfront.net/53130149/j.1749-4486.2009.01962.x20170515-16591-vrj0vj-libre.pdf?1494834830=&response-content-disposition=inline%3B+filename%3DInvestigating_the_thyroid_nodule.pdf&Expires=1729730709&Signature=GYHn~q7fmreYldyp4olzfIgdkcih-CnOOgVv2KvCCgzEnooQp1~JAbIZ6sf9j3C64L2EC0J5i3X2mYbxrjYSO3w-x2I9HIFUI187s-fkOg8dfEOfQCwduGhvu-U5TEBf6EqOdpNtHN0Qq-0mpSzXGJjpp5Pc9tMSaZN06tQ~OGh99KJo1FY0FObXXhPXCT1LEt5QrTaLLaBgN7xB~6RmHD~M1ON7fE3sBlWLCmy~-0kxFVTEbIk42KIGgvasMQOxdqeacFT2X3XbWnACUkWuGzQk3ZIuHeIDGmg37HlCehP8npPHpidV5jy3vo2Q8MwV~nJ3e8XdzTRL3kr3gUvXKA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)


5. **Management of the Solitary Thyroid Nodule**  
   - Authors: Meei J. Yeung, Jonathan W. Serpell  
   - Journal: The Oncologist  
   - Volume: 13, Issue: 2, Pages: 105-112, Year: 2008
   - Link:[Read More](https://theoncologist.onlinelibrary.wiley.com/doi/pdfdirect/10.1634/theoncologist.2007-0212)
 ## License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to customize any sections or add additional information specific to your project! Let me know if you need further assistance.

