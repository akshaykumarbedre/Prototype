import os
import dotenv
import streamlit as st

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = "text-embedding-3-small"

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #0072C6;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 2em;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        .footer {
            position: fixed;
            left: 0; bottom: 0; width: 100%;
            background: #e1eafc;
            color: #333;
            text-align: center;
            padding: 8px 0;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

# Logo and title
col1, col2 = st.columns([1, 8])
with col1:
    # Using an image from Google (credit card/payment image)
    st.image("https://cdn-icons-png.flaticon.com/512/4108/4108843.png", width=64)
with col2:
    st.markdown("<h1 style='color:#0072C6;'>Cashless Card Solutions Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("Ask any question about <b>cashless card solutions</b>.", unsafe_allow_html=True)

st.info("💡 Example questions: \n" + 
    "• How is the smartcard helpful?\n" + 
    "• How does your offline transaction capability work in environments with unreliable internet connectivity?\n" + 
    "• What makes your transaction processing faster than conventional payment methods?\n" + 
    "• Can you explain how the student card functions as a multi-purpose campus tool?\n" + 
    "• What security measures are in place to protect student financial information?\n" + 
    "• How does the end-of-day settlement process work for syncing transaction data?", 
    icon="💡")

sample_question = "How is the smartcard helpful?"

user_query = st.text_input("Enter your question:", value=sample_question)

if st.button("Get Answer"):
    with st.spinner("Getting answer..."):
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings(model=model)
        vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-4o-mini")

        # Custom prompt
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert in providing information about cashless card solutions. Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )

        result = qa_chain({"query": user_query})
        st.success("**Answer:**\n\n" + result["result"])

# Footer
st.markdown('<div class="footer">© 2024 Cashless Card Solutions Chatbot</div>', unsafe_allow_html=True)
