import streamlit as st
import pandas as pd
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
from together import Together

# Initialize TogetherAI API
api_key = '0d69a07d0928aac13d4fb32f80a97a8a4f9f8908db1cec970b1e3a6c6963405a'  # Replace with your actual API key
together_ai = Together(api_key=api_key)

# Function to generate PDF report
def generate_pdf(candidate_details, scores):
    temp_filename = tempfile.mktemp(".pdf")
    pdf_filename = f"{candidate_details['Name']}.pdf"
    c = canvas.Canvas(temp_filename, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 50, "Assessment Report")
    c.drawString(100, height - 100, f"Name: {candidate_details['Name']}")
    c.drawString(100, height - 120, f"Register Number: {candidate_details['Register Number']}")
    c.drawString(100, height - 140, f"College Name: {candidate_details['College Name']}")
    c.drawString(100, height - 160, f"Mail ID: {candidate_details['Mail ID']}")

    c.drawString(100, height - 200, "Scores:")
    for idx, (question_type, score) in enumerate(scores.items()):
        c.drawString(100, height - (220 + 20 * idx), f"{question_type}: {score}")

    c.save()
    return temp_filename, pdf_filename

# Function to evaluate short answer using Together AI
def evaluate_short_answer(question, answer):
    prompt = (f"Evaluate the following answer:\nQuestion: {question}\nAnswer: {answer}\n"
              "Give a score out of 10. Only give me the score as output no other text must be printed or given as response, not even fullstop.")
    response_gen = together_ai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.5,
    )

    response_text = response_gen.choices[0].message.content
    try:
        score = float(response_text)
    except ValueError:
        score = 0.0  # Default score if parsing fails
    return score

# Function to evaluate user code
def evaluate_code_challenge(user_code, test_cases):
    exec_globals = {}
    try:
        # Execute user code in a separate namespace
        exec(user_code, {}, exec_globals)
        # Retrieve the solution function from the executed code
        solution_function = exec_globals.get('solution')
        
        if not callable(solution_function):
            return 0, ["Error: No callable function named 'solution' found."]
        
        passed = 0
        results = []
        for inputs, expected in test_cases:
            try:
                # Unpack inputs and call the function
                user_output = solution_function(*inputs)
                if user_output == expected:
                    passed += 1
                    results.append(f"Pass: Input {inputs} => Output {user_output}")
                else:
                    results.append(f"Fail: Input {inputs} => Expected {expected}, got {user_output}")
            except Exception as e:
                results.append(f"Error during execution: {e}")
        
        return passed, results
    except Exception as e:
        return 0, [f"Error in code: {e}"]

# Streamlit UI
st.title("Assessment Platform Demo")

# Step 1: Candidate Basic Details
if 'candidate_details' not in st.session_state:
    st.session_state['candidate_details'] = {}

if not st.session_state['candidate_details']:
    with st.form("candidate_details_form"):
        name = st.text_input("Name")
        register_number = st.text_input("Register Number")
        college_name = st.text_input("College Name")
        mail_id = st.text_input("Mail ID")
        submitted = st.form_submit_button("Start Test")

        if submitted:
            st.session_state['candidate_details'] = {
                "Name": name,
                "Register Number": register_number,
                "College Name": college_name,
                "Mail ID": mail_id
            }
            st.session_state['show_test_options'] = True

# Step 2: Horizontal Navigation Bar
if 'show_test_options' in st.session_state and st.session_state['show_test_options']:
    st.write(f"Candidate: {st.session_state['candidate_details']['Name']}")
    st.write("**Navigate through the sections below:**")

    section = st.radio(
        "",
        options=['MCQs', 'Short Answers', 'Coding Challenges'],
        index=0,
        horizontal=True,
        key="selected_section"
    )

    # Step 3: Sections for Different Question Types
    if section == 'MCQs':
        if 'mcq_submitted' not in st.session_state:
            uploaded_file = st.file_uploader("Upload CSV with questions", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                questions = df.to_dict(orient='records')
            else:
                questions = [
                    {
                        'question': "What is the primary goal of supervised learning?",
                        'options': [
                            'To identify patterns in data', 
                            'To make predictions based on labeled data', 
                            'To cluster similar data points', 
                            'To optimize a function using gradient descent'
                        ],
                        'answer': 'To make predictions based on labeled data'
                    },
                    {
                        'question': "What is the name of the algorithm that uses a decision tree as a model for classification and regression tasks?",
                        'options': [
                            'Random Forest', 
                            'Support Vector Machine', 
                            'Gradient Boosting', 
                            'CART'
                        ],
                        'answer': 'CART'
                    },
                    {
                        'question': "What is the term for a machine learning model that is trained on a dataset and then deployed to make predictions on new, unseen data?",
                        'options': [
                            'Model evaluation', 
                            'Model selection', 
                            'Model deployment', 
                            'Model serving'
                        ],
                        'answer': 'Model serving'
                    },
                    {
                        'question': "What is the name of the machine learning algorithm that uses a neural network to classify images?",
                        'options': [
                            'Convolutional Neural Network (CNN)', 
                            'Recurrent Neural Network (RNN)', 
                            'Long Short-Term Memory (LSTM)', 
                            'Autoencoder'
                        ],
                        'answer': 'Convolutional Neural Network (CNN)'
                    },
                    {
                        'question': "What is the term for a machine learning model that is trained to minimize the difference between its predictions and the actual values?",
                        'options': [
                            'Mean Absolute Error (MAE)', 
                            'Mean Squared Error (MSE)', 
                            'Root Mean Squared Error (RMSE)', 
                            'Cross-Entropy Loss'
                        ],
                        'answer': 'Root Mean Squared Error (RMSE)'
                    }
                ]
            if st.button("Start MCQ Test"):
                st.session_state['mcq_questions'] = random.sample(questions, min(5, len(questions)))

            if 'mcq_questions' in st.session_state:
                scores = 0
                for i, q in enumerate(st.session_state['mcq_questions']):
                    st.write(f"**Q{i+1}: {q['question']}**")
                    answer = st.radio("", q['options'], key=f"mcq_q{i}")
                    if answer == q['answer']:
                        scores += 1

                if st.button("Submit MCQ Test"):
                    st.session_state['scores'] = st.session_state.get('scores', {})
                    st.session_state['scores']["MCQs"] = scores
                    st.session_state['mcq_submitted'] = True
                    st.success(f"Your score: {scores}/{len(st.session_state['mcq_questions'])}")
        else:
            st.write("MCQ Test already submitted.")

    elif section == 'Short Answers':
        if 'short_answer_submitted' not in st.session_state:
            questions = [
                "Explain the significance of the Turing Test.",
                "What are the main features of Object-Oriented Programming?"
            ]  # Example questions

            selected_question = random.choice(questions)
            st.write(f"**Question: {selected_question}**")
            answer = st.text_area("Your Answer", key="short_answer")

            if st.button("Submit Short Answer"):
                score = evaluate_short_answer(selected_question, answer)
                st.session_state['scores'] = st.session_state.get('scores', {})
                st.session_state['scores']["Short Answers"] = score
                st.session_state['short_answer_submitted'] = True
                st.success(f"Your score: {score}/10")
        else:
            st.write("Short Answer already submitted.")

    elif section == 'Coding Challenges':
        if 'coding_challenge_submitted' not in st.session_state:
            st.write("## Problem Statement")
            st.write("Create a function named `solution` that takes two integers as input and returns their sum.")
            user_code = st.text_area("Write your solution:", height=300)

            test_cases = [
                ((1, 2), 3),
                ((10, -5), 5),
                ((-3, -7), -10),
            ]

            if st.button("Submit Code"):
                if user_code.strip():
                    score, results = evaluate_code_challenge(user_code, test_cases)
                    for result in results:
                        st.write(result)
                    st.session_state['scores'] = st.session_state.get('scores', {})
                    st.session_state['scores']["Coding Challenges"] = score
                    st.session_state['coding_challenge_submitted'] = True
                    st.success(f"Your score: {score}/{len(test_cases)}")
                else:
                    st.error("Please write your code before submitting.")
        else:
            st.write("Coding Challenge already submitted.")

    if 'candidate_details' in st.session_state and 'scores' in st.session_state:
        if st.button("Generate Report"):
            candidate_details = st.session_state['candidate_details']
            scores = st.session_state['scores']
            temp_pdf_path, pdf_filename = generate_pdf(candidate_details, scores)
            with open(temp_pdf_path, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=pdf_filename)
            os.remove(temp_pdf_path)
