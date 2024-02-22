import openai
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def openai_keys():
    openai.api_type = "azure"
    openai.api_base = os.environ.get('API_BASE')
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.environ.get('API_KEY')

    llm = AzureChatOpenAI(
                    openai_api_base=os.environ.get('API_BASE'),
                    openai_api_version="2023-03-15-preview",
                    deployment_name='chatgpt',
                    openai_api_key=os.environ.get('API_KEY'),
                    openai_api_type = "azure",
                    temperature = 0.2,
                    max_tokens=4096,
                    top_p=0.8
                )
    embeddings = OpenAIEmbeddings(openai_api_base=os.environ.get('API_BASE'),
    openai_api_version="2023-03-15-preview",
    openai_api_key=os.environ.get('API_KEY'),
    openai_api_type = "azure",chunk_size=1)

    return openai,llm,embeddings
