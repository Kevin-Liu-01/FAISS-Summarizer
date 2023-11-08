#Using ReduceDocumentsChain
#Simple 
#The refine documents chain constructs a response by looping over the input documents and iteratively updating its answer. 
#For each document, it passes all non-document inputs, the current document, and the latest intermediate answer to an 
#LLM chain to get a new answer.

from dotenv import load_dotenv
import os

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
docs = TextLoader( "summary.txt" ).load()


prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

llm = ChatOpenAI(temperature=0, model_name = "gpt-3.5-turbo-16k")

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    # "Given the new context, refine the original summary in Italian"
    # "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1000, chunk_overlap = 0
    )

split_docs = text_splitter.split_documents(docs)

result = chain({"input_documents": split_docs}, return_only_outputs=True)

print("\n\n".join(result["intermediate_steps"][:3]))
print(result["output_text"])

