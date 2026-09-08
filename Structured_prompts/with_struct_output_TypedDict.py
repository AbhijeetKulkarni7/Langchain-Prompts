from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint 
from dotenv import load_dotenv
from typing import Annotated, TypedDict , Optional

load_dotenv()

llm = HuggingFaceEndpoint(
  repo_id= "meta-llama/Llama-3.1-8B-Instruct" , 
  temperature=0.5,
  max_new_tokens=100, 
  task = "text-generation" 
)

model  = ChatHuggingFace(llm=llm) 

#create a schema for typedDict output 

class Review(TypedDict):

    summary: str
    sentiment: Annotated[str, "return summary either positive negative or neutral"]  
    key_things: Annotated[list[str], "return key things in bullet points"] 
    pros: Optional[str] 
    cons: Optional[str] 

structured_model = model.with_structured_output(Review)

prompt = "The phone is good, the hardware is build with aluminium and camara quality is best. However the software can be beeter and had bugs that need fix."

result  =  structured_model.invoke(prompt)

print(result)
print(type(result))