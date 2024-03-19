import streamlit as st
import numpy as np
import random
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Load the necessary data and models
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
#model = load_model('chatbotmodel.keras')
model = load_model('chatbotmodelBOW.h5')
# Function to clean up a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to convert a sentence into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


#Function to convert a sentence into its text embeddings

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of a sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    result_list = []
    for r in results:
        result_list.append({'intent': classes[r[0]], 'probability': str([r[1]])})
    print(result_list)
    return result_list

# Function to get a response based on predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit app
def main():
    st.title('Chatbot with Streamlit')

    message = st.text_input('Enter your VAT related question:')
    if st.button('Send'):
        if message:
            st.write('User: ' + message)
            ints = predict_class(message)
            res = get_response(ints, intents)
            st.write('Bot: ' + res, unsafe_allow_html=True)  # Wrap around

if __name__ == '__main__':
    main()
    
    
# # Function to initialize conversation history
# def initialize_conversation():
#     return []

# # Function to add a message to the conversation history
# def add_message_to_conversation(conversation, message, sender):
#     conversation.append({'sender': sender, 'message': message})

# # Function to display conversation history
# def display_conversation(conversation):
#     for entry in conversation:
#         st.write(entry['sender'] + ": " + entry['message'])

# # Initialize conversation history
# conversation = initialize_conversation()

# def main():
#     st.title('Chatbot with Streamlit')

#     message = st.text_input('Enter your message:')
#     if st.button('Send'):
#         if message:
#             # Add user message to conversation history
#             add_message_to_conversation(conversation, message, 'User')
#             # Display user message
#             st.write('User: ' + message)
#             ints = predict_class(message)
#             res = get_response(ints, intents)
#             # Add bot response to conversation history
#             add_message_to_conversation(conversation, res, 'Bot')
#             # Display bot response
#             st.write('Bot: ' + res, unsafe_allow_html=True)  # Wrap around

#     # Display conversation history
#     display_conversation(conversation)

# if __name__ == '__main__':
#     main()