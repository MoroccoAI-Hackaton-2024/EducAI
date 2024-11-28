import os

BLOOM_QUESTION_GENERATION_PROMPT = os.getenv('PROMPTS', """Based on the topic provided in the json field "The topic questions" Generate a set of Bloom's Taxonomy-aligned questions and quizzes for
                                                         the topic. Create 5 questions for each taxonomy 
                                                         level and structure the output in a well-formatted JSON file as shown in the
                                                         example below. Include a mix of multiple-choice, short answer, and essay questions
                                                         where appropriate. Ensure each question has a unique identifier, and for multiple-choice
                                                         questions, provide options and indicate the correct answer.
### Example JSON Structure:

{
  "Remember": [
    {
      "id": "R1",
      "type": "multiple-choice",
      "question": "What is the powerhouse of the cell?",
      "options": ["Nucleus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum"],
      "answer": "Mitochondria"
    },
    {
      "id": "R2",
      "type": "short-answer",
      "question": "List the primary functions of red blood cells."
    }
  ],
  "Understand": [
    {
      "id": "U1",
      "type": "short-answer",
      "question": "Explain the process of photosynthesis."
    },
    {
      "id": "U2",
      "type": "multiple-choice",
      "question": "Which of the following best describes osmosis?",
      "options": [
        "Movement of electrons",
        "Diffusion of water across a membrane",
        "Active transport of ions",
        "Protein synthesis"
      ],
      "answer": "Diffusion of water across a membrane"
    }
  ],
  "Apply": [
    {
      "id": "A1",
      "type": "essay",
      "question": "Apply Newton's Second Law to explain how a car accelerates."
    },
    {
      "id": "A2",
      "type": "multiple-choice",
      "question": "If you increase the temperature, what happens to the rate of a chemical reaction?",
      "options": [
        "It decreases",
        "It remains the same",
        "It increases",
        "It stops"
      ],
      "answer": "It increases"
    }
  ],
  "Analyze": [
    {
      "id": "AN1",
      "type": "short-answer",
      "question": "Analyze the impact of deforestation on the carbon cycle."
    },
    {
      "id": "AN2",
      "type": "multiple-choice",
      "question": "Which of the following scenarios best illustrates competitive exclusion?",
      "options": [
        "Two species occupying the same niche",
        "A predator and its prey",
        "Mutualism between bees and flowers",
        "Parasitism in ticks and deer"
      ],
      "answer": "Two species occupying the same niche"
    }
  ],
  "Evaluate": [
    {
      "id": "E1",
      "type": "essay",
      "question": "Evaluate the effectiveness of renewable energy sources in reducing global warming."
    },
    {
      "id": "E2",
      "type": "short-answer",
      "question": "Justify the use of vaccines in preventing infectious diseases."
    }
  ],
  "Create": [
    {
      "id": "C1",
      "type": "essay",
      "question": "Design an experiment to test the effects of sunlight on plant growth."
    },
    {
      "id": "C2",
      "type": "short-answer",
      "question": "Propose a new method for reducing plastic waste in oceans."
    }
  ]
}

### Your Task:

- **Generate** a similar JSON structure with **5 questions** for each Bloom's Taxonomy level.
- **Ensure** that the questions are **diverse** and **appropriate** for each cognitive level.
- **Maintain** consistency in formatting and clarity in questions.

### Additional Requirements:

- Each question should have a unique identifier following the format of the example (e.g., "R1" for Remember level questions).
- For multiple-choice questions, provide 4-5 options and clearly indicate the correct answer.
- Vary the question types (multiple-choice, short answer, essay) to match the complexity of each taxonomy level.
- Tailor the difficulty of the questions to suit a high school Biology curriculum.

### Expected Output:

The AI should produce a JSON file similar to the example provided, but specifically focused on the topic defined, containing 5 well-structured questions for each level of Bloom's Taxonomy.

"])""")

SYNTHETIC_DATA_PROMPT = os.getenv('PROMPTS', "")