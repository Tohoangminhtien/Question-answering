import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf

# Load the model and tokenizer from Hugging Face
model_name = "Tien-THM/QAVi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="tf")

    # Get the model outputs
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning and end of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    answer_end = tf.argmax(answer_end_scores, axis=1).numpy()[0] + 1

    # Convert tokens to answer
    input_ids = inputs["input_ids"].numpy()[0]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer

# Streamlit app layout
st.title("Question Answering with Tien-THM/QAVi Model")
st.write("Provide a question and a context, and the model will give an answer.")

# User input for context and question
context = st.text_area("Context", placeholder="Enter the context here...")
question = st.text_input("Question", placeholder="Enter your question here...")

if st.button("Get Answer"):
    if context and question:
        with st.spinner("Finding the answer..."):
            answer = answer_question(question, context)
            st.write("Answer:", answer)
    else:
        st.write("Please provide both context and question.")
