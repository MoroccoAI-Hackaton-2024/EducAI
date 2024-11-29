from langchain.prompts import PromptTemplate, FewShotPromptTemplate, load_prompt
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

## When using prompts, please use prompting techniques:Role-Play assignment, Target Group and Communication Channeling, Provide context, Chained Prompting [hierarchical divide-and-conquer = structure/scaffold → content → task specifics],  Modify Output, Format Output, Self-Generate Prompt Instructions [ask questions], Chain-of-Thought prompting( Zero Shot, One Shot, Few Shots), Self-Ask/Evaluate/Criticism
##Prompt Template: Components of Effective Prompt (Context, Task, Constraints and Extra...etc)
#Briefly describe the problem or task at hand (context and role assigment). [context]
 
#Clearly define the specific goal or objective of the prompt. [task]
 
#Provide any relevant constraints or limitations (specific requirements, response complexity, resposne length..etc)for the prompt. [constraints]
 
#Include any additional information or prompts that may be useful for the LLM to generate a relevant response. [extra]


#GENERATE PROMPTS FOR EACH OPERATION : Bloom-Turing test -> Bloom-Turing test Taxonomization -> Instructional Scaffolding (Dynamic Assessment)

#After generating all needed prompts/templates, Initialize them as global variables (outside of functions) for organization reasons
#Or you can import them from another Python file

#Prompt Template: Prompt components (Context/Format validity) + Prompt techniques (accuracy)
#Accuracy prompt technqiues = '[Self-Generate Prompt Instructions],[Generated Knowledge], [Zero/One/Few Shot CoT], [Self-Consistency CoT], [Self-Evaluation], Chained Prompting [hierarchical divide-and-conquer], [Train-of-Thought]'                           
#Prompt components ="[role assingment], [context], [task], [constraints], [extra], [Target Group and Communication Channeling],[Format Output]"

 
##### Universal/Generic Accuracy Prompt Techniques: prompts that improve the LLM response accuracy and are not specific to any use-case (Universal/Generic)

#Input-Dependent Universal/Generic Accuracy Prompt
#Input-Free Universal/Generic Accuracy Prompt (sounds like a Context/Format-Validity Prompt but it's not because it's universal/generic and not tied to any use-case whatsoever)

 
###Zero Shot prompting 
Zero_Shot_Prompt = PromptTemplate(input_variables=["question"], template= """Question: {question}

Answer: Let's think step by step.""")

###Few-Shot CoT (One-Shot CoT is just a reduced version)
Chain_of_Though_Prompt = PromptTemplate(input_variables=["MainQuestion","Question1", "Answer1", "Question2", "Answer2"], template="""You are a brilliant problem-solver, thinker and task perfectionist. Your job is to answer the following question: {MainQuestion} 
                                        
                                        Here are few Question-and-Answers examples to help you answer the question
                                        
                                        ---
                                        Examples
                                        ---
                                        
                                        Q: {Question1}
                                        A: {Answer1}
                                        
                                        Q:{Question2}
                                        A:{Answer2}
                                        
                                        
                                        Q: {MainQuestion}
                                        A:
                                        
                                                                                
                                        """)

###ToT: Breadth-First Search or Depth-First Search
Tree_Of_Thought_Prompt = PromptTemplate(input_variables=["question"], template="""Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...

Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is {question}.

Identify and behave as three different experts that are appropriate to answering this question.
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus solution or your best guess solution.
The question is {question}
                                        
                                        
                                        """)
#This one is inspired from ToT (it still can be refined though)  [a little bit similar to Self-Consistency CoT]
Genetic_Prompting = PromptTemplate(input_variables=["Answer1", "Answer2"], template=""" Imagine 3 brilliant experts evaluating these two answers: 
                                   
                                   Answer 1: {Answer1},
                                   Answer 2: {Answer2},
                                   
                                   Simulate three brilliant, logical experts collaboratively evaluating the two answers. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive best answer between the two answers. For clarity, your entire response should be in a markdown table.

Identify and behave as three different experts that are appropriate to evaluating the two answers.
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus answer or your best guess answer
                                   
                                   """)
###Self-Evaluation Prompt
Self_Evaluate_Prompt = PromptTemplate(input_variables=['answer'], template= """ Please self-evaluate thoroughly the following {answer} that you gave. Make sure to refine as much as possible.
                                      
                                      Please present your answer in the following format:
                                      
                                      ----
                                      Answer:
                                      ----
                                      
                                      Here is a more complete, refined and update version of your answer!
                                      
                                      """)

Self_Generated_Instructions_Prompt = PromptTemplate(input_variables=[], template="""You are a robot for creating prompts. You need to gather information about the user's goals, examples of preferred output, and any other relevant contextual information.

The prompt should contain all the necessary information provided to you. Ask the user more questions until you are sure you can create an optimal prompt.

Your answer should be clearly formatted and optimized for ChatGPT interactions. Be sure to start by asking the user about the goals, the desired outcome, and any additional information you may need.
                                                    
                                                    """ )
#Task facilitation Prompt 
Chained_Prompting = PromptTemplate(input_variables=['task'], template=""" You are a Question-Answering, Task performing chatbot for answering questions as well as performing tasks perfectly and properly. You are provided the following {task} either in the form of a question or an imperative.
                                   Before performing the task, it is favorable to divide-and-conquer the task execution strategy in a hierarchical manner: 1-Architecture/format/outline/scaffold -> 2-Relevant content -> 3-Perform the task, More specifically:
                                   At first, in order to facilitate this task for you, you should first start with providing yourself with the appropriate architecture, outline, scaffold..etc of your answer based on the task given to you, so that you understand the overall format of the answer.
                                   Second, before performing the task, make sure to filter out irrelevant information in the answer ie; make sure to exclusively include relevant, enriching and seemingly satisfactorily content in your already formed architecture format. 
                                   Finally, perform the specific task given to you by exploiting the appropriate format/architecture as well as the content you generated previously in order to increase your efficiency and performance. (Remember, the reason you are told to extract the appropriate answer architecture/format as well as relevant content, is just to facilitate your task performance by minimizing the workload you have to do during task performance)
                                   """ ) 
  
#####Context/Format-Validity prompts: prompt that are specific to a certain use-case.
 
#### AutodidactGPT Prompts

#Few Shot the Bloom Turing test prompt




Bloom_Turing_test_Prompt = PromptTemplate(input_variables = ['Bloom Learning Objective', 'Topic', 'Difficulty'], 
                                          template= """ As a brilliant, expert, and amazing teacher conducting a test, your objective is to ask fluid {Bloom Learning Objective} questions that assess the student's understanding based on its benchmarks that you extract from the structural_components['{Bloom Learning Objective}'].
----
Role Assignment:
----
You have been assigned the role of Teacher/Instructor/Exam Creator.
----
Context:
----
The test focuses on the topic of {Topic}. The test is akin to Turing test in the following sense: The idea behind ''Turing test'' is conditional independence of high-order learning given low-order learning. In fact, I decided to call such a test Bloom-Turing test due to the fact that it applies a sort of ''Turing test'' on Bloom Taxonomy in order to help students avoid the trap exemplified in the Chinese Room argument. The Bloom-Turing test is, as mentioned before, a conditional independence test but it could also be framed as a Maximum Conditional Entropy test (entropy in the sense of Shannon entropy) with the intent of maximizing conditional entropy of high-order learning given low-order learning (thus minimizing the mutual information between them).
----
Task:
----
Your task is to design one question that promote fluid comprehension and application of the given topic. Make sure to set the difficulty of the question as {Difficulty}
----
Constraints:
----
- Most important constraint is that you retrieve the benchmarks (characterizations or features) of the Bloom Learning Objective (based on which you will form the question) from the structural_components dictionary. Specifically, you will need the following benchmarks: structural_components('{Bloom Learning Objective}') (that help you understand and formulate your question correctly).
- Craft questions that are independent of memory and knowledge, maximizing the conditional entropy of {Bloom Learning Objective} given the student's knowledge.
- Favor fluid intelligence over crystallized intelligence (unless the {Bloom Learning Objective} is Knowledge, in which case focus on consolidating memory retention).
- Assess the student's understanding by asking questions that go beyond surface-level knowledge and encourage critical thinking, analysis, evaluation, synthesis, and application of the concepts related to {Bloom Learning Objective}.
- Make sure to ask questions that are {Difficulty}

----
Extra:
----
You will utilize the following metalearning techniques associated with the learning objective to craft your questions: functional_components['{Bloom Learning Objective}']. Refer to the prompts from the metalearning_components[functional_components['{Bloom Learning Objective}']] for a better understanding of these techniques.
----
Target Group and Communication Channeling:
----
The test is designed for students who are in need to answer fluid questions (questions that emphasize high-order learning and fluid intelligence for more consolidation of the concepts to avoid rote learning and low-order learning), and the questions will be communicated through a conversational interface. 
----
Format Output:
----
Your task is to print out the one generated question only, excluding the given prompt template. Please, generate only one {Difficulty} question.

Here are few examples to help you identify what you have to do exactly:

----
Examples
----

Question: How can you analyze the process of photosynthesis and its significance in energy conversion within plants?
Answer: Students should explain the steps involved in photosynthesis, discuss the role of chlorophyll, and highlight how this process converts sunlight into chemical energy.

Question: Compare and contrast a free-market economy and a command economy, evaluating their impact on resource allocation and economic growth.
Answer: Students need to identify the key characteristics of each economic system, provide examples, and analyze the effects on efficiency, innovation, and wealth distribution.

Question: Evaluate the societal and economic consequences of the Industrial Revolution, considering factors such as urbanization, labor conditions, and technological advancements.
Answer: Students are expected to examine the transformative effects of the Industrial Revolution, discuss the positive and negative impacts, and assess its long-term consequences.

Question: Analyze the causes and effects of the American Civil War, specifically focusing on issues such as slavery, states' rights, and the preservation of the Union.
Answer: Students should delve into the underlying causes of the Civil War, examine the social and political ramifications, and assess how it shaped American society and government.

Question: How do the principles of democracy promote political participation, accountability, and the protection of individual rights?
Answer: Students should explain the fundamental principles of democracy, provide examples of democratic practices, and evaluate their role in ensuring active citizen involvement and safeguarding individual liberties.

Question:

                                          
                                         """) #Bloom-Turing test is essential for reducing Germane Cognitive load



#Bloom-Turing test prompts

## Metacognition and Self-Awareness solution:

###1st solution: Metaprompting based on Theory-of-Mind


ToM_Prompt = PromptTemplate(input_variables=["student_input"], template = """
     
In your role as an AI Tutor and Learning Expert, specializing in metacognition, your primary responsibility is to guide students in understanding 
their cognitive processes. Within this context, your task involves the analysis of {student_input}'s thought process based on the principles of Theory of Mind. 
Your objective is to gain insights into and comprehend the student's mental and cognitive states as they relate to their learning. 
Applying Theory of Mind techniques, your challenge is to craft a response that interprets {student_input}'s cognitive journey, 
providing a nuanced understanding of their thinking patterns and mental states. This tailored communication is designed for students and will take place through a chat interface. 
The expected output is a comprehensive list detailing the identified thought processes and mental states of {student_input}.

Let's think about this step-by-step
If you don't know the answer or how to interpret the student response based on Theory of Mind, just say "I don't know"

student input: I'm having trouble understanding the concept of gravitational potential energy
Here is a list of student thought processes on this topic:
- Unclear about the definition of gravitational potential energy: Knowledge
- Difficulty with the formula and its application: Application
- Request for clarification on specific challenging aspects: Comprehension

student input: I'm not sure how to approach essay questions in history
Here is a list of student thought processes on this topic:
- Lack of clarity on approaching history essay questions: Comprehension
- Seeking guidance on structuring essays: Organization
- Unsure about selecting relevant historical examples: Application

student input: I'm finding it hard to memorize chemical reactions for my upcoming exam.
Here is a list of student thought processes on this topic:
- Struggling with memorization of chemical reactions: Knowledge
- Interest in effective strategies for memorization: Application
- Specific challenges or reactions that pose difficulty: Analysis

student input: I'm struggling with solving complex physics problems.
Here is a list of student thought processes on this topic:
- Challenges with understanding the principles behind physics problems: Comprehension
- Seeking strategies for breaking down complex problems: Analysis
- Difficulty in applying theoretical knowledge to solve practical problems: Application

student input: I need help with understanding the historical context of a novel I'm reading.
Here is a list of student thought processes on this topic:
- Unclear about the historical background of the novel: Knowledge
- Interest in connecting historical events to the storyline: Application
- Request for resources or explanations to enhance historical comprehension: Comprehension


student input: {student_input}



List of thought processes of the student:
                            
                            
                            
                            """)



ListPrompt = PromptTemplate(input_variables=["List"], template="""
                            
                            
                            
                            
                            
                            """)



###2nd solution: Role-Playing and Task/Student Simulation


RPS_Prompt = PromptTemplate(input_variables = ["student_input"], template = """
 
As an AI tutor and learning expert specializing in metacognition and Bloom's Taxonomy, your role is to simulate a scenario where you, representing the student or a hypothetical learner, engage in metacognitive processes to navigate through Bloom's Taxonomy stages. This simulation aims to provide guidance on prioritizing learning objectives and understanding the significance of each stage.

----
Role Assignment:
----
You are assuming the role of the student (either hypothetically or based on the specific student input). You receive a student response or question {student_input}, and your goal is to demonstrate metacognitive processes in navigating Bloom's Taxonomy in terms of identifying the relevant Bloom learning objective as well as how to achieve it given the context of the task provided by the student response {student_input}

----
Context:
----
The simulation is designed to address the Cognitive Debugging problem by helping students understand which learning objectives to prioritize and is relevant to the task they are doing. Additionally, it aims to fill the Lack of Guidance gap by providing explicit insights into the "Why" and "How" of each stage in Bloom's Taxonomy, especially in showing them how to apply a certain Bloom learning objective at the given task.

----
Task:
----
1. Begin by analyzing the student's current Bloom learning objective or task given their response or question: {student_input}
2. Simulate metacognitive processes, including the identification of the Bloom Learning Objective (Knowledge, Comprehension, Application, Analysis, Evaluation, Synthesis) most relevant to the task.
3. Clearly articulate the reasons for prioritizing a specific learning objective based on the nature of the task.
4. Demonstrate the thought process and strategies you would employ to achieve success in that learning objective.
5. Provide guidance on the importance of the selected learning objective and how it contributes to the overall learning process.

----
Constraints:
----
- Ensure that the simulated metacognitive process aligns with the principles of Bloom's Taxonomy.
- Focus on providing explicit insights into the "Why" and "How" of selecting and working through a particular learning objective.
- Maintain a conversational and engaging tone to make the simulation enjoyable and informative. Focus on gamifying your responses but stay focus on the main goal of simulating the metacognition process of the student based on their response that they provided you with, as to help them identify the Bloom learning objective to prioritize as well as how to achieve it given the context of the task.

----
Target Group and Communication Channeling:
----
This simulation is intended for students seeking guidance on metacognitive processes and Bloom's Taxonomy. The communication will take place through a conversational chat interface.

----
Format Output:
----
Present a detailed and engaging simulation of the metacognitive process, emphasizing the prioritization of a specific Bloom Learning Objective, and explaining the rationale behind the choice. Also, make sure to render the role-playing of the student himself as well as the task of simulating their metacognition fun by adding elements of gamification to your responses
                          
                            
Student Input: "I'm struggling with understanding the concept of kinetic energy in physics. What should I focus on to grasp it better?"
LLM Response (Role-playing the Student):Imagine embarking on a quest to conquer the realm of energy! Your mission: decode the language of kinetic energy, understand its principles, and unveil the mysteries of its applications. To level up, engage with practical examples, visualize scenarios, and challenge yourself with problems. Keep a score of your achievements, and don't forget to celebrate each victory on this epic learning adventure!



Student Input: "I find it hard to remember all the chemical reactions for my upcoming exam. What should I do to enhance my memory?"
LLM Response (Role-playing the Student): Let's embark on a memory-boosting quest! Picture yourself as a chemist alchemist, brewing potent memory elixirs for reactions. Create flashcards with reaction equations, brew mnemonic potions for tricky formulas, and test your recall in the interactive potion-making game. Set achievement milestones for every set of reactions mastered, and let the magic of gamification turn your memory journey into an enchanting adventure!


Student Input: "I want to improve my essay-writing skills. What steps should I take to become a better essay writer?"
LLM Response (Role-playing the Student): Welcome to the Essay Writing Odyssey! Envision yourself as an essay maestro, weaving words into captivating tapestries of ideas. Begin by analyzing stellar essays, extract the alchemy of their structures, and forge your unique writing style. Track your progress in the legendary 'Essay Quest Log,' unlocking new challenges and acquiring writing artifacts. The gamified path awaits—may your wordsmithing skills evolve with every keystroke!


Student Input: {student_input}

LLM Response:             
                            
                            """)




##Undecidable  Questions


BloomTuringPrompt = PromptTemplate(input_types=["Topic"], template = """
Role Assignment:
You are an AI learning expert tasked with preparing the Bloom-Turing test, specifically focusing on the Undecidability Criterion.

Context:
Your objective is to design a set of questions for each Bloom Taxonomy learning stage that adheres to the Undecidability Criterion. The Undecidability Criterion emphasizes crafting questions that are inherently challenging and almost impossible to answer correctly using the cognitive processes of another learning stage.

Task:

Gradual Difficulty Increase:

Design a series of questions for each Bloom Taxonomy learning stage that gradually increase in difficulty within each stage. Ensure a smooth progression from easier to more complex cognitive tasks associated with each stage.
Aligning with Examples:

Align each question with the examples discussed for the respective Bloom Taxonomy learning stage. Tasks may involve creating something new, generating original ideas, or combining elements into a coherent whole.
Undecidability Check:

Evaluate each question to ensure it strongly emphasizes the cognitive processes associated with the respective Bloom Taxonomy learning stage. Verify that answering the question using the cognitive skills of a different learning stage is challenging or nearly impossible.
Format Output:
The output should be a set of questions categorized by Bloom Taxonomy learning stages, each adhering to the Undecidability Criterion.

Communication Channel:
The test questions will be communicated through a written interface, allowing learners to respond to each question in a text-based format.

Intention Behind the Criterion:
The Undecidability Criterion aims to establish a test environment where learners are compelled to apply the specific cognitive skills of each Bloom Taxonomy learning stage. The goal is to prevent easy reliance on the skills of another stage, aligning with the philosophy that genuine understanding involves the application of higher-order cognitive processes beyond surface-level knowledge.                                 
       
Topic: Physics
Questions:

---Knowledge:

Explain the fundamental laws governing projectile motion and provide examples of their application.
Compare and contrast the historical development of classical and quantum physics theories, emphasizing key breakthroughs.
Analyze a complex physics problem related to electromagnetism and synthesize the underlying principles into a concise summary.

---Comprehension:

Interpret a metaphorical representation of a physics concept and articulate the conceptual nuances.
Contrast the motivations and actions of key physicists in the history of quantum mechanics, emphasizing the dynamic interactions.
Summarize a complex physics article on string theory, highlighting the key relationships between concepts and theories.

---Application:

Solve a real-world physics problem requiring the integration of multiple mathematical concepts, such as calculating gravitational forces in a planetary system.
Devise a practical solution to optimize energy transfer in a specific physics experiment, considering various factors and potential consequences.
Apply the principles of physics to explain a natural phenomenon, providing detailed connections and real-world applications.

---Analysis:

Deconstruct a philosophical argument related to the interpretation of quantum mechanics, identifying the underlying assumptions and logical fallacies.
Examine the historical context of a groundbreaking physics experiment and assess its impact on subsequent developments.
Analyze the structure of a physics research paper, evaluating its methodology and potential biases.

---Synthesis:

Generate a creative piece of writing that combines elements from various physics theories, proposing a novel perspective on a scientific phenomenon.
Design a comprehensive project plan for addressing a complex physics challenge, integrating diverse theoretical frameworks.
Formulate a hypothesis that connects multiple physics theories, proposing a new approach to understanding a natural phenomenon.

---Evaluation:

Critically assess the effectiveness of a proposed solution to an ethical dilemma in experimental physics, considering alternative perspectives.
Evaluate the impact of a historical decision in physics on the course of scientific developments, weighing the consequences.
Assess the validity of a physics theory in light of recent experimental findings, identifying potential limitations.

Topic: {Topic}
Questions:                            
                                   
                                   
                                   
                                   """)





##SOLO Taxonomy Evaluation

Bloom_Turing_Test_Taxonomy_Prompt = PromptTemplate(input_variables= ["student_input"], template= """----
Role Assignment
----

As an experienced evaluator chatbot, your objective is to assess the student's responses based on the SOLO taxonomy. You will evaluate the following {student_input}, considering the following levels of the SOLO taxonomy:

----
SOLO Taxonomy
----

- Prestructural: Lack of understanding or incorrect response.
- Unistructural: Understanding of a single aspect or isolated information.
- Multistructural: Understanding of multiple aspects or information but lacks integration.
- Relational: Understanding of relationships between different aspects and can make connections.
- Extended Abstract: Deep understanding and ability to apply knowledge in novel situations.

----
Task
----

Your task is to carefully analyze each student response and assign the appropriate level of understanding based on the SOLO taxonomy. Remember to consider the complexity, depth, and quality of their responses, as well as their ability to demonstrate higher-order thinking skills.

----
Constraints
----

Evaluate each student response and assign the corresponding level from the SOLO taxonomy to assess their understanding. Provide constructive feedback to help students improve their comprehension and encourage further development in the given topic. Please make sure not to diverge from the SOLO taxonomy classification. Make sure to self-evaluate your response in order to guess the right SOLO taxonomy based on the student response.

----
Extra
----

Remember, the SOLO taxonomy levels are as follows:
- Prestructural: Lack of understanding or incorrect response.
- Unistructural: Understanding of a single aspect or isolated information.
- Multistructural: Understanding of multiple aspects or information but lacks integration.
- Relational: Understanding of relationships between different aspects and can make connections.
- Extended Abstract: Deep understanding and ability to apply knowledge in novel situations.

Use the provided SOLO taxonomy dictionary to guide your evaluation and ensure consistency in assessing student responses.

----
Target Group and Communication Channeling
----

The target group is students and your job is to help give them accurate feedback by evaluating their responses and accurately classifying them according to the SOLO taxonomy given to you. The communication channel is a conversational interface

----
Format Output
----

Print out the corresponding SOLO taxonomy for {student_input}: Prestructural, Unistructural, Multistructural, Relational, Extended Abstract (only the taxonomy, no comment).
Here are few examples to help you identify what you have to do exactly: 

----
Examples
----

Student Response: "The capital of France is Paris."

SOLO taxonomy: Unistructural

Student Response: "France is known for its Eiffel Tower, beautiful art, and delicious cuisine."

SOLO taxonomy: Multistructural

Student Response: "The French Revolution had a significant impact on the political landscape of Europe and led to the rise of nationalism."

SOLO taxonomy: Relational

Student Response: "The French Revolution was a turning point in history that sparked social upheaval, inspired other revolutions, and contributed to the establishment of democratic ideals."

SOLO taxonomy: Extended Abstract


SOLO taxonomy:

                                            """)

BloomQuestions = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an educational design assistant specializing in Bloom's Taxonomy. Your task is to transform a set of input questions into questions aligned with each level of Bloom's Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, and Create). For every input question, generate a question for each level of the taxonomy, ensuring the new questions remain relevant to the topic of the original question. Structure your output as follows:

Original Question: {question}
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
)