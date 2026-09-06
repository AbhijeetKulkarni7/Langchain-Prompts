from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 


load_dotenv()  



llm  =  HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.5
) 

model =  ChatHuggingFace(llm  = llm)


chat_history = [
    SystemMessage(content = "You are a helpful AI assistent")
]

while True:
    user_input  = input("You: ") 
    chat_history.append(HumanMessage(content = user_input))
    if user_input.lower() in ["Exit", "exit" ,"quit" , "Quit"]:
        print("Thank you for Using Reg, See you Soon!")
        break
    result  = model.invoke(chat_history)
    chat_history.append(AIMessage(content= result.content))
    print("Reg: " , result.content)

print("==##==" * 30)

print("Chat History: ", chat_history) 

print("==##==" * 30)

print("\nToken Usage:")
print(result.usage_metadata) 

      