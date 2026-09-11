from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv 
import os 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv() 

## Step 1: Create Prompt Template 

prompt =  PromptTemplate(
    template="Generate 5 intresting unknown fact about topic: {topic}",
    input_variables=["topic"] 

)

## Define model 

llm  =  HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.6, 
)

model  =  ChatHuggingFace(
    llm = llm,  
    prompt = prompt 
)

## step 3 :  Define parser 
parser  =  StrOutputParser() 

## create a chain 

chain  =  prompt | model  | parser 

result =  chain.invoke({"topic":"Football"})

print("Starting the simple chain....\n ")
print("**" * 20 , "\n")
print(result)

## Visulaize the chain  

chain.get_graph().print_ascii()