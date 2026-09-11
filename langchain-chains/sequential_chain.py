from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv 
import os 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv() 



prompt1 =  PromptTemplate(
    template="Generate a detailed report topic: {topic}",
    input_variables=["topic"] 

)

prompt2 =  PromptTemplate(
    template="Generate a 5 point summary on topic: {text}",
    input_variables=["text"] 

)

## Define model 

llm  =  HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.6, 
)

model  =  ChatHuggingFace(
    llm = llm
)

## step 3 :  Define parser 
parser  =  StrOutputParser() 


chain  =  prompt1 | model  | parser | prompt2 | model | parser

result  =  chain.invoke({"topic": "India 3 stage Nuclear Plan"})

print(result)



