from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv 
import os 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableParallel

load_dotenv() 




## Define model 
## As I am using free open source model 
## maintaining only one LLM , If you can affort use ChatGPT & Claude of two diffrent models.  
llm  =  HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation",
    temperature = 0.6, 
)

model1  =  ChatHuggingFace(
    llm = llm
)

model2  =  ChatHuggingFace(
    llm = llm
)

# Step 2 : Prompt 


prompt1 =  PromptTemplate(
    template="Generate short and simple notes from following text \n : {text}",
    input_variables=["text"] 

)

prompt2 =  PromptTemplate(
    template="Generate a 5 questions list from following \n : {text}",
    input_variables=["text"] 

)
prompt3 =  PromptTemplate(
    template="merge the Notes and Questions into single document \n notes -> {notes} & quiz ->  {quiz}",
    input_variables=["notes" , "quiz"] 

)

## step 3 :  Define parser 
parser  =  StrOutputParser() 

parallel_chain  = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)

merge_chains =  prompt3 | model1 | parser 

chain  =  parallel_chain | merge_chains 

result =  chain.invoke({"text" : "Logistic Regression"})
print(result )

chain.get_graph().print_ascii()