import random
import json 
import pickle # serializing and de-serializing a Python object structure.
import numpy as np


import nltk # natural language toolkit
from nltk.stem import WordNetLemmatizer # The WordNetLemmatizer is specifically used for 
#lemmatization.Lemmatization is the process of reducing words to their base or root form, which is known as the lemma

#neural network imports
from tensorflow.keras.models import Sequential # The Sequential class in Keras is a linear stack of layers, and it's a convenient way to build a deep learning model layer by layer.
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
#The Dense layer represents a fully connected layer in a neural network.
# The Activation layer applies an activation function element-wise to the output of the previous layer. 
# Activation functions introduce non-linearity to the model, allowing it to learn complex patterns.
#The Dropout layer is used for regularization in neural networks. It randomly sets a fraction of input units to zero at each update during training time, which helps prevent overfitting by reducing the reliance on specific neurons. This, in turn, promotes the learning of more robust features.

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# In the context of neural networks and deep learning, optimization algorithms like SGD are used to minimize the loss function during the training process.

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = [] # stores word tokens 
classes = [] #  stores the intent tags (tags)
documents = [] # (tuples of word lists and intent tags)
ignore_letters = ['.', '!', ',', '?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) # tokenize the pattern into a list of words
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
#save the words anc class objects into a pickle file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#represent words as numerical values, use bag of words
training = []
output_empty = [0] * len(classes)

#to store the length of train_x = 
len_x = len(words)
print(len_x, 'len_x')
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in  words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    bag.extend(output_row)
    training.append(bag)

random.shuffle(training)
#print(*training[:len(training)//2], sep = '\n')
training = np.array(training)


train_x = list(training[:, :len_x])
train_y = list(training[:, len_x:])


# train_x = np.array([i[0] for i in training])
# train_y = np.array([i[1] for i in training])

#create the Neural Network

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))



##################################
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())  # Add batch normalization layer
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))



#################################



lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9
)

# Use the learning rate schedule in the optimizer
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)


# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#stochastic gradient descent
# breakpoint()
# sgd = SGD(learning_rate=0.01, decay = 1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metric = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 500,batch_size = 50, verbose = 1, callbacks=[early_stopping])
#model.save('chatbotmodel.keras', hist)
model.save('chatbotmodelBOW.h5', hist)
print('Done')
