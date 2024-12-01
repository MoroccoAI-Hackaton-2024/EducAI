import streamlit as st
import tempfile
import os
import json
from dotenv import load_dotenv
import keys  # Ensure this module contains your OpenAI API key as `key`
import llm  # Ensure this module contains the QueryRunner class
import datetime  # For timestamping saved files
import agents  # Import your prompts from agents.py

# Load environment variables if needed
load_dotenv()

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = keys.key

# Import prompts from agents.py
TAXONOMY_AGENT_PROMPT = agents.TAXONOMY_AGENT_PROMPT
SCORING_AGENT_PROMPT = agents.SCORING_AGENT_PROMPT
METACOGNITION_AGENT_PROMPT = agents.METACOGNITION_AGENT_PROMPT  # Updated agent prompt

# =====================================
# UI Configuration Variables
# =====================================

# Colors
BACKGROUND_COLOR = "#c3c1b4"  # White background
BUTTON_COLOR = "#F19CBB"      # Blue button color
TEXT_COLOR = "#000000"        # Black text

# Images
LOGO_IMAGE_LEFT = "../Images/hackatonai_logo.png"
LOGO_IMAGE_RIGHT = "../Images/educai-logo.png"

# Texts
APP_TITLE = "Learnify"
ANSWERS_SUBMITTED_MESSAGE = "✅ Your answers have been submitted and saved!"
STUDENT_SCORED_MESSAGE = "✅ The student's performance has been scored!"
METACOGNITIVE_SUCCESS_MESSAGE = "✅ Metacognitive recommendations generated successfully!"

st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    h1 {
        color: #333333;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0d6efd;
        border-radius: 10px;
        border: 1px solid #0d6efd;
    }
    .reportview-container .main .block-container{
        padding-top: 5rem;
        padding-left: 5%;
        padding-right: 5%;
    }
</style>
""", unsafe_allow_html=True)



# =====================================
# Initialize Session State for storing answers and JSON data
# =====================================
if 'file_content' not in st.session_state:
    st.session_state['file_content'] = None
if 'answers' not in st.session_state:
    st.session_state['answers'] = {}
if 'transformed_questions' not in st.session_state:
    st.session_state['transformed_questions'] = None
if 'json_response' not in st.session_state:
    st.session_state['json_response'] = None
if 'restructured_data' not in st.session_state:
    st.session_state['restructured_data'] = None
if 'restructured_filename' not in st.session_state:
    st.session_state['restructured_filename'] = None
if 'original_filename' not in st.session_state:
    st.session_state['original_filename'] = None
if 'answers_submitted' not in st.session_state:
    st.session_state['answers_submitted'] = False
if 'scored_data' not in st.session_state:
    st.session_state['scored_data'] = None
if 'scored_filename' not in st.session_state:
    st.session_state['scored_filename'] = None
if 'weights' not in st.session_state:
    st.session_state['weights'] = {}
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = None
if 'taxonomy_evaluation' not in st.session_state:
    st.session_state['taxonomy_evaluation'] = None  # New entry for Taxonomy-Based Evaluation

# Define Taxonomy Levels
taxonomy_levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


# Streamlit App Title and Logos
def display_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(LOGO_IMAGE_RIGHT, width=100)  # Adjust the width as needed
    with col2:
        st.markdown("""
        <h1 style='margin-top: 20px;'>Learnify</h1>
        <p style='color: #4a4a4a; font-size: 24px;'>Tailored Learning, Empowering Success.</p>
        """, unsafe_allow_html=True)

st.sidebar.markdown("""
## Explore
Navigate through the sections to experience personalized learning.
""")

# Upload File and Read Content
def upload_file():
    st.write("""
        Upload a text file containing your questions. The app will transform each question into questions aligned with each level of Bloom's Taxonomy.
    """)
    uploaded_file = st.file_uploader("Drag and drop a file here, or click to select a file", type=["txt"])
    if uploaded_file is not None:
        st.write(f"**Uploaded File:** {uploaded_file.name}")
        file_content = uploaded_file.read().decode("utf-8")
        with st.expander("📄 View Uploaded File Content"):
            st.text(file_content)
        st.session_state['file_content'] = file_content  # Store file content in session state
        return True
    else:
        st.info("Please upload a `.txt` file to get started.")
        return False

# Run LLM Query for Taxonomy Agent
def run_llm_query(TAXONOMY_AGENT_PROMPT):
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", etc.
    file_content = st.session_state['file_content']
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt") as tmp_file:
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    try:
        bloom_taxonomy_agent = llm.QueryRunner(document_path=temp_file_path, model_name=MODEL_NAME)
        bloom_taxonomy = bloom_taxonomy_agent.run_query(TAXONOMY_AGENT_PROMPT)
        result_str = bloom_taxonomy.get('result', '').strip()

        if result_str:
            # Remove code block markers if present
            if result_str.startswith("```json") and result_str.endswith("```"):
                result_str = result_str[7:-3].strip()
            elif result_str.startswith("```") and result_str.endswith("```"):
                result_str = result_str[3:-3].strip()

            # Try parsing the JSON
            try:
                json_response = json.loads(result_str)
                st.session_state['json_response'] = json_response  # Store in session state
                st.session_state['transformed_questions'] = json_response.get("Topic Questions", [])
                return True
            except json.JSONDecodeError as e:
                st.error(f"JSON decoding failed: {e}")
                st.write("LLM Response was:", result_str)
                st.write("Please ensure that the input file is correctly formatted and that the LLM response is valid JSON.")
                return False
        else:
            st.error("No response from the language model.")
            return False
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Display Questions and Collect Answers
def display_questions_and_collect_answers():
    transformed_questions = st.session_state['transformed_questions']
    if transformed_questions:
        st.subheader("Transformed Questions Aligned with Bloom's Taxonomy")
        with st.form(key='answer_form'):
            for idx, question_set in enumerate(transformed_questions):
                original_question = question_set.get("Original Question", f"Question {idx+1}")
                st.markdown(f"**Original Question {idx+1}:** {original_question}")
                for level in taxonomy_levels:
                    taxonomy_question = question_set.get(level, "N/A")
                    if taxonomy_question == "N/A":
                        continue  # Skip if the taxonomy question is not available
                    answer_field = f"{level} Answer_{idx}"
                    st.markdown(f"### {level}")
                    st.write(taxonomy_question)
                    default_value = st.session_state['answers'].get(answer_field, "")
                    answer = st.text_area(
                        label=f"Your Answer for {level}:",
                        value=default_value,
                        key=answer_field,
                        height=100
                    )
                    st.session_state['answers'][answer_field] = answer
                st.markdown("---")
            submit_button = st.form_submit_button(label='Submit Your Answers')
        if submit_button:
            st.session_state['answers_submitted'] = True
        return submit_button
    else:
        st.error("No transformed questions to display.")
        return False

# Restructure JSON as per the new format
def restructure_json():
    transformed_questions = st.session_state['transformed_questions']
    restructured_data = {"Bloom Taxonomy": {level: [] for level in taxonomy_levels}}

    for idx, question_set in enumerate(transformed_questions):
        original_question = question_set.get("Original Question", f"Question {idx+1}")
        for level in taxonomy_levels:
            taxonomy_question = question_set.get(level, "N/A")
            if taxonomy_question == "N/A":
                continue  # Skip if the taxonomy question is not available
            answer_field = f"{level} Answer_{idx}"
            student_answer = st.session_state['answers'].get(answer_field, "")
            sub_question = {
                "Original Question": original_question,
                "Sub-Question": {
                    "Question": taxonomy_question,
                    "Answer": student_answer
                }
            }
            restructured_data["Bloom Taxonomy"][level].append(sub_question)
    st.session_state['restructured_data'] = restructured_data
    return restructured_data

# Save JSON File Locally
def save_json_file(data, filename_prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return filename
    except Exception as e:
        st.error(f"Failed to save JSON file: {e}")
        return None

# Run LLM Query for Scoring Agent
def run_scoring_agent():
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", etc.

    restructured_data = st.session_state['restructured_data']
    # Convert restructured_data to JSON string
    input_json = json.dumps(restructured_data, ensure_ascii=False, indent=4)

    # Prepare the prompt with the input JSON
    scoring_prompt = SCORING_AGENT_PROMPT.replace("{input_json}", input_json)

    # Write the input JSON to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".json") as tmp_file:
        tmp_file.write(input_json)
        temp_file_path = tmp_file.name

    try:
        scoring_agent = llm.QueryRunner(document_path=temp_file_path, model_name=MODEL_NAME)
        scoring_response = scoring_agent.run_query(scoring_prompt)
        result_str = scoring_response.get('result', '').strip()

        if result_str:
            # Remove code block markers if present
            if result_str.startswith("```json") and result_str.endswith("```"):
                result_str = result_str[7:-3].strip()
            elif result_str.startswith("```") and result_str.endswith("```"):
                result_str = result_str[3:-3].strip()

            # Try parsing the JSON
            try:
                scored_data = json.loads(result_str)
                st.session_state['scored_data'] = scored_data  # Store in session state
                return True
            except json.JSONDecodeError as e:
                st.error(f"JSON decoding failed: {e}")
                st.write("LLM Response was:", result_str)
                st.write("Please ensure that the LLM response is valid JSON.")
                return False
        else:
            st.error("No response from the language model.")
            return False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Calculate Taxonomy-Based Evaluation and Store JSON
def calculate_taxonomy_evaluation():
    scored_data = st.session_state['scored_data']
    taxonomy_evaluation = {"Bloom Taxonomy": {}}

    for level in taxonomy_levels:
        if level in scored_data["Bloom Taxonomy"]:
            total_score = 0
            count = 0
            for sub_question in scored_data["Bloom Taxonomy"][level]:
                score = sub_question["Sub-Question"].get("score", 0)
                total_score += score
                count += 1
            average_score = total_score / count if count > 0 else 0
            weight = st.session_state['weights'].get(level, 0)
            weighted_average = average_score * weight
            taxonomy_evaluation["Bloom Taxonomy"][level] = {
                "average_score": average_score,
                "weight": weight,
                "weighted_average": weighted_average
            }
        else:
            taxonomy_evaluation["Bloom Taxonomy"][level] = {
                "average_score": 0,
                "weight": st.session_state['weights'].get(level, 0),
                "weighted_average": 0
            }

    st.session_state['taxonomy_evaluation'] = taxonomy_evaluation
    return taxonomy_evaluation

# Run LLM Query for Metacognitive Recommendation Agent
def run_metacognition_agent():
    MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", etc.

    # Retrieve the Taxonomy-Based Evaluation JSON from session state
    taxonomy_evaluation = st.session_state.get('taxonomy_evaluation')

    if not taxonomy_evaluation:
        st.error("Taxonomy-Based Evaluation data is missing.")
        return False

    # Display the taxonomy_evaluation
    st.markdown("#### Taxonomy-Based Evaluation JSON:")
    st.json(taxonomy_evaluation)

    # Convert the taxonomy_evaluation to a JSON string
    input_json_str = json.dumps(taxonomy_evaluation, ensure_ascii=False, indent=4)
    print("#############################")
    print("input_json_str:")
    print(input_json_str)

    # Display the input_json_str
    st.markdown("#### Input JSON String:")
    st.code(input_json_str, language='json')

    # Display the METACOGNITION_AGENT_PROMPT before replacement
    st.markdown("#### METACOGNITION_AGENT_PROMPT before replacement:")
    st.code(METACOGNITION_AGENT_PROMPT)

    # Check for the placeholder in the prompt
    placeholder = "{input_json}"
    if placeholder in METACOGNITION_AGENT_PROMPT:
        print("Placeholder '{input_json}' found in METACOGNITION_AGENT_PROMPT")
    else:
        print("Placeholder '{input_json}' NOT found in METACOGNITION_AGENT_PROMPT")
        st.error("Placeholder '{input_json}' not found in METACOGNITION_AGENT_PROMPT.")
        return False

    # Prepare the prompt with the input JSON
    # Use .format() method
    metacognition_prompt = METACOGNITION_AGENT_PROMPT.format(input_json=input_json_str)
    print("metacognition_prompt:")
    print(metacognition_prompt)

    # Display the metacognition_prompt
    st.markdown("#### Metacognition Agent Prompt:")
    st.code(metacognition_prompt)

    # Write the prompt to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt") as tmp_file:
        tmp_file.write(metacognition_prompt)
        temp_file_path = tmp_file.name

    try:
        print(f"Temporary file created at: {temp_file_path}")

        # Initialize QueryRunner with document_path and model_name
        metacognition_agent = llm.QueryRunner(document_path=temp_file_path, model_name=MODEL_NAME)

        # Run the query
        metacognition_response = metacognition_agent.run_query(metacognition_prompt)
        result_str = metacognition_response.get('result', '').strip()

        # Display the agent's response
        st.markdown("#### Metacognition Agent Response:")
        st.code(result_str)

        print("Metacognition Agent Response:")
        print(result_str)

        if result_str:
            # Store the recommendations
            st.session_state['recommendations'] = result_str
            return True
        else:
            st.error("No response from the language model.")
            return False

    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        return False

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted.")

# Display Question-Based Evaluation
def display_question_based_evaluation():
    scored_data = st.session_state['scored_data']
    levels_present = [level for level in taxonomy_levels if level in scored_data["Bloom Taxonomy"]]
    num_questions = max(len(scored_data["Bloom Taxonomy"][level]) for level in levels_present) if levels_present else 0
    questions = [f"Question {i+1}" for i in range(num_questions)]

    # Let the user select a question
    selected_question = st.selectbox("Select a question to evaluate:", questions)

    # Get the index of the selected question
    idx = questions.index(selected_question)

    # Display the scores for each taxonomy level for the selected question
    st.markdown(f"### Evaluation for {selected_question}")
    total_weighted_score = 0.0
    for level in taxonomy_levels:
        if level in scored_data["Bloom Taxonomy"] and idx < len(scored_data["Bloom Taxonomy"][level]):
            sub_question = scored_data["Bloom Taxonomy"][level][idx]
            score = sub_question["Sub-Question"].get("score", 0)
            weight = st.session_state['weights'].get(level, 0)
            adjusted_score = score * weight
            total_weighted_score += adjusted_score
            # Display the score using custom HTML bars
            st.write(f"**{level}**")
            percentage = int((score / 5.0) * 100)
            bar_color = "#76c7c0"  # Customize the color
            bar_html = f"""
            <div style="background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px;">
                <div style="width: {percentage}%; background-color: {bar_color}; height: 100%; border-radius: 5px;"></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Score: {score}/5")
            with col2:
                st.write(f"Weight: {weight:.2f}")
            with col3:
                st.write(f"Weighted Score: {adjusted_score:.2f}")
            st.markdown("---")
        else:
            st.warning(f"No data for level '{level}' in {selected_question}.")
    st.write(f"**Total Weighted Score for {selected_question}: {total_weighted_score:.2f} out of 5.00**")

# Display Taxonomy-Based Evaluation
def display_taxonomy_based_evaluation():
    scored_data = st.session_state['scored_data']
    levels_present = [level for level in taxonomy_levels if level in scored_data["Bloom Taxonomy"]]
    num_questions = max(len(scored_data["Bloom Taxonomy"][level]) for level in levels_present) if levels_present else 0

    st.markdown("### Taxonomy-Based Evaluation")
    total_weighted_score = 0.0
    taxonomy_evaluation = {"Bloom Taxonomy": {}}

    for level in taxonomy_levels:
        if level in scored_data["Bloom Taxonomy"]:
            total_score = 0
            level_questions = scored_data["Bloom Taxonomy"][level]
            for sub_question in level_questions:
                score = sub_question["Sub-Question"].get("score", 0)
                total_score += score
            average_score = total_score / len(level_questions) if level_questions else 0
            weight = st.session_state['weights'].get(level, 0)
            weighted_average = average_score * weight
            total_weighted_score += weighted_average

            taxonomy_evaluation["Bloom Taxonomy"][level] = {
                "average_score": average_score,
                "weight": weight,
                "weighted_average": weighted_average
            }

            # Display the score using custom HTML bars
            st.write(f"**{level}**")
            percentage = int((average_score / 5.0) * 100)
            bar_color = "#76c7c0"  # Customize the color
            bar_html = f"""
            <div style="background-color: #e0e0e0; border-radius: 5px; width: 100%; height: 20px;">
                <div style="width: {percentage}%; background-color: {bar_color}; height: 100%; border-radius: 5px;"></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Average Score: {average_score:.2f}/5")
            with col2:
                st.write(f"Weight: {weight:.2f}")
            with col3:
                st.write(f"Weighted Avg Score: {weighted_average:.2f}")
            st.markdown("---")
        else:
            st.warning(f"Level '{level}' is missing in scored data.")
    st.write(f"**Total Weighted Average Score: {total_weighted_score:.2f} out of 5.00**")

    # Store the taxonomy evaluation for metacognition
    st.session_state['taxonomy_evaluation'] = taxonomy_evaluation

# Display Metacognitive Recommendations
def display_metacognitive_recommendations():
    recommendations = st.session_state['recommendations']
    st.markdown("### Metacognitive Recommendations")
    st.write(recommendations)

    # Optionally, save the recommendations to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metacognitive_recommendations_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(recommendations)

    # Download button for recommendations
    st.download_button(
        label="📥 Download Recommendations",
        data=recommendations,
        file_name=filename,
        mime='text/plain'
    )

# Main App Logic
def main():
    display_header()

    # Step 1: File upload
    if st.session_state['file_content'] is None:
        if not upload_file():
            return  # Stop execution until file is uploaded

    # Step 2: Run LLM Query
    if st.session_state['transformed_questions'] is None:
        if not run_llm_query(TAXONOMY_AGENT_PROMPT):
            return  # Stop execution if there's an error

    # Step 3: Display Questions and Collect Answers
    if not st.session_state['answers_submitted']:
        if not display_questions_and_collect_answers():
            return  # Wait until answers are submitted

    # Step 4: Restructure JSON and Save Files
    if st.session_state['restructured_data'] is None:
        restructure_json()

    if st.session_state['restructured_filename'] is None:
        restructured_filename = save_json_file(st.session_state['restructured_data'], "taxonomy_bloom_structured")
        st.session_state['restructured_filename'] = restructured_filename

    if st.session_state['original_filename'] is None:
        original_filename = save_json_file({"Topic Questions": st.session_state['transformed_questions']}, "transformed_questions")
        st.session_state['original_filename'] = original_filename

    # Display success message and options
    st.success(ANSWERS_SUBMITTED_MESSAGE)
    # Display Download Buttons and JSON View Options
    st.markdown("### Additional Features")
    # Display options
    with st.expander("📊 Display Restructured Bloom's Taxonomy JSON"):
        st.json(st.session_state['restructured_data'])
    with st.expander("📄 Display Original Transformed JSON"):
        st.json({"Topic Questions": st.session_state['transformed_questions']})
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Restructured JSON",
            data=json.dumps(st.session_state['restructured_data'], ensure_ascii=False, indent=4),
            file_name=st.session_state['restructured_filename'],
            mime='application/json'
        )
    with col2:
        st.download_button(
            label="📥 Download Original Transformed JSON",
            data=json.dumps({"Topic Questions": st.session_state['transformed_questions']}, ensure_ascii=False, indent=4),
            file_name=st.session_state['original_filename'],
            mime='application/json'
        )

    # Step 5: Scoring
    st.markdown("### Scoring")
    score_button = st.button('Score the Student Performance')
    if score_button:
        if run_scoring_agent():
            scored_filename = save_json_file(st.session_state['scored_data'], "student_score")
            st.session_state['scored_filename'] = scored_filename
            st.success(STUDENT_SCORED_MESSAGE)
            # Calculate taxonomy evaluation
            calculate_taxonomy_evaluation()

    if st.session_state.get('scored_data'):
        # Display Scored Data
        with st.expander("📊 Display Scored Data"):
            st.json(st.session_state['scored_data'])
        # Download Scored Data
        st.download_button(
            label="📥 Download Scored JSON",
            data=json.dumps(st.session_state['scored_data'], ensure_ascii=False, indent=4),
            file_name=st.session_state['scored_filename'],
            mime='application/json'
        )

        # Evaluation Options
        st.markdown("### Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Show Question-Based Evaluation'):
                display_question_based_evaluation()
        with col2:
            if st.button('Show Taxonomy-Based Evaluation'):
                display_taxonomy_based_evaluation()

        # Step 6: Metacognitive Recommendations
        st.markdown("### Metacognitive Recommendations")
        recommend_button = st.button('Get Metacognitive Recommendations')
        if recommend_button:
            if run_metacognition_agent():
                st.success(METACOGNITIVE_SUCCESS_MESSAGE)

        if st.session_state.get('recommendations'):
            display_metacognitive_recommendations()

    # Sidebar Configuration for Weights
    st.sidebar.header("Configuration")
    st.sidebar.subheader("Set Weights for Taxonomy Levels")
    default_weights = [1/6]*6  # Default equal weights if none provided
    if not st.session_state['weights']:
        st.session_state['weights'] = dict(zip(taxonomy_levels, default_weights))

    weights_input = {}
    total_weight = 0.0
    for level in taxonomy_levels:
        weight = st.sidebar.number_input(f"Weight for {level}", min_value=0.0, max_value=1.0, value=st.session_state['weights'][level], step=0.05)
        weights_input[level] = weight
        total_weight += weight

    # Normalize weights if total_weight != 1
    if total_weight != 1.0 and total_weight > 0:
        st.sidebar.warning("Weights do not sum to 1. They will be normalized automatically.")
        for level in taxonomy_levels:
            st.session_state['weights'][level] = weights_input[level] / total_weight
    elif total_weight == 0:
        st.sidebar.error("Total weight cannot be zero. Resetting to default weights.")
        st.session_state['weights'] = dict(zip(taxonomy_levels, default_weights))
    else:
        st.sidebar.success("Weights sum to 1.")
        st.session_state['weights'] = weights_input.copy()

if __name__ == "__main__":
    main()
