import streamlit as st
import tempfile
import os
import json
import string
from dotenv import load_dotenv
import keys  # Ensure this module contains your OpenAI API key as `key`
import llm  # Ensure this module contains the QueryRunner class

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
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", etc.
    MAX_TOKENS = 500
    TOP_N_CHUNKS = 3

    # Define the query template
    query = """
    You are an educational design assistant specializing in Bloom's Taxonomy. Your task is to transform a set of input questions into questions aligned with each level of Bloom's Taxonomy: Remember, Understand, Apply, Analyze, Evaluate, and Create.

    For each input question, generate corresponding questions at each taxonomy level, ensuring they are relevant to the original question's topic.

    **Your output should be a well-formatted JSON object with a single key "Topic Questions" that maps to an array of question objects. Each question object must include the following keys: "Original Question", "Remember", "Understand", "Apply", "Analyze", "Evaluate", and "Create", with their respective aligned questions as string values.**

    **Ensure the JSON is valid and free from any additional text, code blocks, or formatting. Do not include numeric keys within the array.**

    **Example Output:**

    ```json
    {
        "Topic Questions": [
            {
                "Original Question": "What is photosynthesis?",
                "Remember": "Define photosynthesis.",
                "Understand": "Explain how photosynthesis works.",
                "Apply": "Describe how photosynthesis affects plant growth.",
                "Analyze": "Compare photosynthesis and cellular respiration.",
                "Evaluate": "Assess the importance of photosynthesis in ecosystems.",
                "Create": "Design an experiment to measure the rate of photosynthesis."
            },
            {
                "Original Question": "How does gravity affect planetary orbits?",
                "Remember": "What is gravity?",
                "Understand": "Explain the role of gravity in planetary orbits.",
                "Apply": "Calculate the gravitational force between Earth and the Moon.",
                "Analyze": "Compare the effects of gravity on different planetary orbits.",
                "Evaluate": "Evaluate the impact of gravity on the stability of the solar system.",
                "Create": "Propose a model to demonstrate gravity's effect on orbiting bodies."
            }
        ]
    }
    ```

    **Input Questions:**

    ```
    {{context}}
    ```
    """

    # Initialize the QueryRunner with the uploaded file content
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt") as tmp_file:
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    try:
        # Initialize the QueryRunner with the temporary file path and model name
        query_runner = llm.QueryRunner(document_path=temp_file_path, model_name=MODEL_NAME)

        # Run the query
        response = query_runner.run_query(query)

        # **Access Only the 'result' Field**
        result_str = response.get('result', '')

        if result_str:
            # **Clean the JSON String**
            # Remove code block markers if present
            if result_str.startswith("```json") and result_str.endswith("```"):
                result_str = result_str.replace("```json", "").replace("```", "").strip()

            # Remove any non-printable characters
            def clean_string(s):
                printable = set(string.printable)
                return ''.join(filter(lambda x: x in printable, s))

            result_str = clean_string(result_str)

            # **Parse the JSON**
            try:
                json_response = json.loads(result_str)
            except json.JSONDecodeError as e:
                st.error(f"JSON decoding failed: {e}")
                st.write("Please ensure that the input file is correctly formatted.")
                json_response = None

            # **Process the Parsed JSON**
            if isinstance(json_response, dict) and "Topic Questions" in json_response:
                transformed_questions = json_response["Topic Questions"]
            elif isinstance(json_response, list):
                transformed_questions = json_response
            else:
                st.error("Unexpected JSON structure.")
                transformed_questions = []

            if transformed_questions:
                # **Transformed Questions Section**
                st.subheader("Transformed Questions Aligned with Bloom's Taxonomy")

                # Iterate over each question set and create a button
                for idx, question_set in enumerate(transformed_questions):
                    original_question = question_set.get("Original Question", f"Question {idx+1}")
                    # Truncate the question for the button label if it's too long
                    button_label = f"Question {idx+1}: {original_question[:50]}..." if len(original_question) > 50 else f"Question {idx+1}: {original_question}"

                    # Create a unique key for each button to avoid Streamlit's duplicate key error
                    if st.button(button_label, key=f"btn_{idx}"):
                        # Display the aligned questions when the button is clicked
                        st.markdown(f"**Original Question:** {original_question}")

                        # Display each level of Bloom's Taxonomy
                        for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
                            taxonomy_question = question_set.get(level, "N/A")
                            st.markdown(f"### {level}")
                            st.write(taxonomy_question)
            else:
                st.error("No transformed questions available to display.")
        else:
            st.error("No 'result' field found in the response.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
else:
    st.info("Please upload a `.txt` file to get started.")

# Optional: Display additional information or settings
st.sidebar.header("Configuration")
# You can add more configuration options here if needed