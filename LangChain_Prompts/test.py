from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Explain what a transformer model is in simple terms.")

print(result.content)