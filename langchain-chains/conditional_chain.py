from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate   
from dotenv import load_dotenv 
import os 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv() 


llm  =  HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "texxt-generation"
)

model  = ChatHuggingFace(llm  = llm )

parser  =  StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of feedback')

parser2  = PydanticOutputParser(pydantic_object=Feedback)

prompt1 =  PromptTemplate(
    template="Classify the sentiment of following into positive or negative. \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain =  prompt1 | model  | parser2 

# result=  classifier_chain.invoke({'feedback' :'This is very bad shoe.'})
# print(result)  # The sentiment is **Negative**. The word "bad" is a clear indicator of a negative sentiment.


prompt2 =  PromptTemplate(
    template="Write an appropriate feedback to this positive feedback. \n {feedback}",
    input_variables=['feedback']
)

prompt3 =  PromptTemplate(
    template="Write an appropriate feedback to this negative feedback. \n {feedback}",
    input_variables=['feedback']
)

branch_chain =  RunnableBranch(
    (lambda x:x.sentiment == 'positive' , prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative' , prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
 )

chain  =  classifier_chain | branch_chain 

result  =  chain.invoke({"feedback" : "This is super product. Must try once!!"})

print(result)

chain.get_graph().print_ascii()