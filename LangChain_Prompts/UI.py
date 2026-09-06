from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


# Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=500,
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)


# Streamlit UI
st.header("📚 Research Paper Summarizer")

user_input = st.text_area(
    "Enter your research paper or prompt:",
    height=300
)


if st.button("Summarize"):

    if user_input:

        result = model.invoke(
            f"""
            You are an expert research paper summarizer.

            Summarize the following research paper clearly.

            Include:

            1. Research Problem
            2. Motivation
            3. Proposed Methodology
            4. Dataset
            5. Key Results
            6. Main Contributions
            7. Limitations
            8. Future Work

            Do not invent information that is not present
            in the provided text.

            Research Paper:
            {user_input}
            """
        )

        st.subheader("Summary")

        st.write(result.content)

    else:
        st.warning("Please enter the research paper.")