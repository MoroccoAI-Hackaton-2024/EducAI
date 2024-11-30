import streamlit as st
from typing import List
import tempfile
import keys  # Ensure this module contains your OpenAI API key as `key`
import os
from dotenv import load_dotenv
import llm  # Ensure this module contains the QueryRunner class
import json

# Load environment variables if needed
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = keys.key

# Streamlit App Title
st.title("Bloom's Taxonomy Question Transformer")

# Instructions
st.write("""
    Upload a text file containing your questions. The app will transform each question into questions aligned with each level of Bloom's Taxonomy.
""")

# File Uploader
uploaded_file = st.file_uploader("Drag and drop a file here, or click to select a file", type=["txt"])

if uploaded_file is not None:
    # Display the uploaded file name
    st.write(f"**Uploaded File:** {uploaded_file.name}")

    # Read the file content
    file_content = uploaded_file.read().decode("utf-8")

    # Optionally, display the content
    with st.expander("📄 View Uploaded File Content"):
        st.text(file_content)

    # Define other configuration parameters
    # You can also make these configurable via Streamlit widgets if needed
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", etc.
    MAX_TOKENS = 500
    TOP_N_CHUNKS = 3

    # Define the query template
    query = f"""
    You are an educational design assistant specializing in Bloom's Taxonomy. Your task is to transform a set of input questions into questions aligned with each level of Bloom's Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, and Create). For every input question, generate a question for each level of the taxonomy, ensuring the new questions remain relevant to the topic of the original question. Structure your output as follows:

    Original Question: {{context}}
    Remember: [Question targeting recall of knowledge]
    Understand: [Question requiring comprehension]
    Apply: [Question involving practical application]
    Analyze: [Question prompting breakdown into components]
    Evaluate: [Question asking for judgment or critique]
    Create: [Question requiring synthesis or creation of new ideas]
    Example:
    Input Question: What are the causes of climate change?

    Remember: What is climate change, and what are its primary causes?
    Understand: How do greenhouse gases contribute to climate change?
    Apply: Can you identify the greenhouse gas emissions in your daily activities?
    Analyze: What are the key differences between natural and human-induced causes of climate change?
    Evaluate: How effective are current policies in mitigating climate change?
    Create: Propose a new strategy to reduce the effects of climate change on urban areas
    """

    # Initialize the QueryRunner with the uploaded file content
    # Assuming QueryRunner can accept file content directly. If it only accepts file paths, use a temporary file.
    # Here, we'll use a temporary file to simulate the original behavior.

    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt") as tmp_file:
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    try:
        # Initialize the QueryRunner with the temporary file path and model name
        query_runner = llm.QueryRunner(document_path=temp_file_path, model_name=MODEL_NAME)

        # Run the query
        response = query_runner.run_query(query)

        # Check if the response is a JSON-serializable object
        try:
            json_response = json.loads(response) if isinstance(response, str) else response
        except json.JSONDecodeError:
            st.error("The response from the model is not valid JSON.")
            st.text(response)
            json_response = None

        if json_response:
            # Pretty-print the JSON response using Streamlit's JSON viewer
            st.subheader("Transformed Questions Aligned with Bloom's Taxonomy")
            st.json(json_response)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
else:
    st.info("Please upload a `.txt` file to get started.")

# Optional: Display additional information or settings
st.sidebar.header("Configuration")
# You can add more configuration options here if needed
