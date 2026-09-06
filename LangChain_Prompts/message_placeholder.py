from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder


chat_prompt = ChatPromptTemplate(
    messages = [
    SystemMessage(content = "You are a helpful assistant"), 
    MessagesPlaceholder(variable_name = "chat_history"),    
    HumanMessage(content = "{query}")
    ]
)

chat_history = [] 

with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())



prompt  = chat_prompt.invoke({'chat_history': chat_history, 'query':'What is Langchain?'})

print("Empty file")

print(chat_history)

print(prompt)
