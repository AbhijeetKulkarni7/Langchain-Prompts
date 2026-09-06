from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from dotenv import load_dotenv 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

load_dotenv() 



llm =  HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="text-generation", 
    temperature=0.5, 
    max_new_tokens=100
)

model  =  ChatHuggingFace(llm =llm)

messages = [ 
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "Tell me about langchain"),
]

result  = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print("==##==" * 30) 

print(messages)

print("==##==" * 30) 

print("\nToken Usage:")
print(result.usage_metadata)


