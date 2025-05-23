import os
import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

# ------------------------------
# Load secrets from Streamlit
# ------------------------------
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_CHAT_DEPLOYMENT_NAME = st.secrets["AZURE_CHAT_DEPLOYMENT_NAME"]
AZURE_CHAT_API_VERSION = st.secrets["AZURE_CHAT_API_VERSION"]
AZURE_EMBEDDING_DEPLOYMENT_NAME = st.secrets["AZURE_EMBEDDING_DEPLOYMENT_NAME"]
AZURE_EMBEDDING_API_VERSION = st.secrets["AZURE_EMBEDDING_API_VERSION"]

ASTRA_DB_TOKEN = st.secrets["ASTRA_DB_TOKEN"]
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
ASTRA_DB_COLLECTIONS = st.secrets["ASTRA_DB_COLLECTIONS"]



# ------------------------------
# Set Azure environment variables for embeddings
# ------------------------------
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT
os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_EMBEDDING_API_VERSION

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("VISION :violet[360]")

selected_collection = st.selectbox("Select a collection", ["All"] + ASTRA_DB_COLLECTIONS)
user_input = st.text_area("Ask your question:", placeholder="e.g. Tell me about Roche")
submit = st.button("Submit")

# ------------------------------
# LangChain Components
# ------------------------------

# Embeddings (uses env vars)
embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_EMBEDDING_DEPLOYMENT_NAME,
    chunk_size=1000
)

# Chat Model (explicit API version)
llm = AzureChatOpenAI(
    deployment_name=AZURE_CHAT_DEPLOYMENT_NAME,
    api_version=AZURE_CHAT_API_VERSION,
    temperature=1
)

# Create retrievers for each collection
retrievers = {}
for coll in ASTRA_DB_COLLECTIONS:
    retrievers[coll] = AstraDBVectorStore(
        embedding=embeddings,
        collection_name=coll,
        token=ASTRA_DB_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    ).as_retriever()

# ------------------------------
# Handle Query
# ------------------------------
if submit:
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        try:
            if selected_collection == "All":
                st.info("Searching across all collections...")
                results = []
                for name, retriever in retrievers.items():
                    chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=False,
                        chain_type="stuff"
                    )
                    answer = chain.run(user_input)
                    results.append((name, answer))

                for name, answer in results:
                    st.subheader(f"Collection: {name}")
                    st.write(answer)

            else:
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retrievers[selected_collection],
                    return_source_documents=False,
                    chain_type="stuff"
                )
                response = chain.run(user_input)
                st.subheader(f"Response from '{selected_collection}':")
                st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")
