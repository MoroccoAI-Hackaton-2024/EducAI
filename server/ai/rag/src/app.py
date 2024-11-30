import streamlit as st
import tempfile
import os
import json
from dotenv import load_dotenv
import keys  # Ensure this module contains your OpenAI API key as `key`
import llm  # Ensure this module contains the QueryRunner class
import datetime  # For timestamping saved files

# Load environment variables if needed
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = keys.key

# Initialize Session State for storing answers
if 'answers' not in st.session_state:
    st.session_state['answers'] = {}
    
# Create two columns
col1, col2 = st.columns([1,1])

# Place the first logo in the first column
with col1:
    st.image("../Images/hackatonai_logo.png", use_container_width=True)

# Place the second logo in the second column
with col2:
    st.image("../Images/educai-logo.png", use_container_width=True)


# Streamlit App Title
st.title("Learnify")

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

        # Access Only the 'result' Field
        result_str = response.get('result', '')

        if result_str:
            # Clean the JSON String
            # Remove code block markers if present
            if result_str.startswith("```json") and result_str.endswith("```"):
                result_str = result_str.replace("```json", "").replace("```", "").strip()

            # Parse the JSON
            try:
                json_response = json.loads(result_str)
            except json.JSONDecodeError as e:
                st.error(f"JSON decoding failed: {e}")
                st.write("Please ensure that the input file is correctly formatted.")
                json_response = None

            # Display Raw JSON
            if json_response:
                with st.expander("📄 View Raw JSON Output"):
                    st.json(json_response)

                # Process the Parsed JSON
                if isinstance(json_response, dict) and "Topic Questions" in json_response:
                    transformed_questions = json_response["Topic Questions"]
                elif isinstance(json_response, list):
                    transformed_questions = json_response
                else:
                    st.error("Unexpected JSON structure.")
                    transformed_questions = []

                if transformed_questions:
                    # Transformed Questions Section
                    st.subheader("Transformed Questions Aligned with Bloom's Taxonomy")

                    # Create a form for the answers
                    with st.form(key='answer_form'):
                        # Iterate over each question set and display taxonomy-aligned questions with answer fields
                        for idx, question_set in enumerate(transformed_questions):
                            original_question = question_set.get("Original Question", f"Question {idx+1}")

                            # Display Original Question
                            st.markdown(f"**Original Question {idx+1}:** {original_question}")

                            # Iterate over each taxonomy level and display the question with a text area for answers
                            for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
                                taxonomy_question = question_set.get(level, "N/A")
                                answer_field = f"{level} Answer_{idx}"  # Unique key per question and level

                                st.markdown(f"### {level}")
                                st.write(taxonomy_question)

                                # Display the text area for the student's answer
                                # Pre-fill with existing answer if available
                                if st.session_state['answers'].get(answer_field):
                                    default_value = st.session_state['answers'][answer_field]
                                else:
                                    default_value = ""

                                answer = st.text_area(
                                    label=f"Your Answer for {level}:",
                                    value=default_value,
                                    key=answer_field,
                                    height=100
                                )

                                # Update the session state with the new answer
                                st.session_state['answers'][answer_field] = answer

                            # Add a horizontal line to separate different questions
                            st.markdown("---")

                        # Submit Button
                        submit_button = st.form_submit_button(label='Submit Your Answers')

                    if submit_button:
                        # Collect all answers from session state
                        student_answers = {"Topic Questions": []}

                        for idx, question_set in enumerate(transformed_questions):
                            original_question = question_set.get("Original Question", f"Question {idx+1}")
                            answer_set = {
                                "Original Question": original_question,
                                "Sub-Questions": {}
                            }

                            for level in ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]:
                                taxonomy_question = question_set.get(level, "N/A")
                                answer_field = f"{level} Answer_{idx}"  # Unique key per question and level
                                student_answer = st.session_state['answers'].get(answer_field, "")

                                answer_set["Sub-Questions"][level] = {
                                    "Question": taxonomy_question,
                                    "Answer": student_answer
                                }

                            student_answers["Topic Questions"].append(answer_set)

                        # Timestamp for unique filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        answers_file_path = os.path.join(os.getcwd(), f"student_answers_{timestamp}.json")

                        try:
                            # Save the JSON file locally
                            with open(answers_file_path, 'w', encoding='utf-8') as f:
                                json.dump(student_answers, f, ensure_ascii=False, indent=4)

                            # Optional: Provide a download button for the JSON file
                            json_str = json.dumps(student_answers, ensure_ascii=False, indent=4)
                            st.download_button(
                                label="📥 Download Your Answers",
                                data=json_str,
                                file_name=f'student_answers_{timestamp}.json',
                                mime='application/json'
                            )

                            # Display success message
                            st.success("✅ Your answers are submitted!!")

                        except Exception as e:
                            st.error(f"Failed to save answers: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
else:
    st.info("Please upload a `.txt` file to get started.")

# Optional: Display additional information or settings
st.sidebar.header("Configuration")
# You can add more configuration options here if needed
