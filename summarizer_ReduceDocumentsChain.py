#Using ReduceDocumentsChain
#For this, we'll first map each document to an individual summary using an LLMChain. 
# Then we'll use a ReduceDocumentsChain to combine those summaries into a single global summary.
# The map reduce documents chain first applies an LLM chain to each document individually (the Map step), 
# treating the chain output as a new document. It then passes all the new documents to a separate combine 
# documents chain to get a single output (the Reduce step). It can optionally first compress, or collapse, 
# the mapped documents to make sure that they fit in the combine documents chain (which will often pass them to an LLM). 
# This compression step is performed recursively if necessary.



from dotenv import load_dotenv
import os

from langchain.chains import (
    StuffDocumentsChain, LLMChain, ReduceDocumentsChain
)
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import TextLoader

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

nTokens = 4000

docs = TextLoader( "summary.txt" ).load()

# map prompt which remain unchanged
map_prompt = PromptTemplate.from_template("""
Summarize the following text in a clear and concise way:
TEXT:"{text}"
Brief Summary:
""")
          
llm = ChatOpenAI(temperature=0, model_name = "gpt-3.5-turbo-16k")
map_chain = LLMChain(llm=llm, prompt=map_prompt)

reduce_prompts = ["""
Generate a summary of the following text that includes the following elements:

Text:"{text}"
""", """
You are Markdown writer. Generate a summary of the following text:

Text:"{text}"
"""]

# loop over reduce prompts
for promptText in reduce_prompts:

    reduce_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template( promptText ) )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="text"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max= nTokens ,
    )

    # map_chain could provide a pre computed output, since docs and map_prompt remain unchanged
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="text",
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1000, chunk_overlap = 0
    )

    split_docs = text_splitter.split_documents(docs)
    text = map_reduce_chain.run(split_docs) 
    print(text)
