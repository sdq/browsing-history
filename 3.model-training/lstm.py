from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence
import numpy as np
import gensim
import pickle

model = gensim.models.Word2Vec.load("model")
max_sentence_length = 0

f = open('Train.txt', 'r')
lines = f.readlines()
X = []
Y = []
for line in lines:
    line = line[:-1]
    sentence = line.split("\t")
    vectors = []
    urls = sentence[:-1]
    url_count = len(urls)
    if url_count > max_sentence_length:
        max_sentence_length = url_count
    # print("url_count:%s" % url_count)
    for url in urls:
        vectors.append(model[url])
    # print("sum:%s" % vector)
    if url_count == 0:
        continue
    # vector = np.true_divide(vector, url_count)
    # print("divide:%s" % vector)
    # print(X)
    # if len(vector) != 100:
    #     print("wrong")
    X.append(vectors)
    label = [0,0,0]
    if sentence[-1] == "0":
        label = [1,0,0]
    elif sentence[-1] == "1":
        label = [0,1,0]
    elif sentence[-1] == "2":
        label = [0,0,1]
    else:
        print("wrong")
    Y.append(label)

X_train = sequence.pad_sequences(X, maxlen=max_sentence_length)
# X_train = X_train[:-1138]
# Y_train = Y[:-1138]
# X_val = X_train[-1120:]
# Y_val = Y[-1120:]
X_train = X_train[:-18]
Y_train = Y[:-18]


####
f = open('Test.txt', 'r')
sentences = []
lines = f.readlines()
lines_count = len(lines)
right_count = 0
X_test = []
Y_test = []
for line in lines:
    line = line[:-1]
    sentence = line.split("\t")
    sentences.append(sentence)

    vectors = []
    urls = sentence[:-1]
    url_count = len(urls)
    # print("url_count:%s" % url_count)
    if url_count == 0:
        continue
    for url in urls:
        vectors.append(model[url])
    X_test.append(vectors)

    label = [0,0,0]
    if sentence[-1] == "0":
        label = [1,0,0]
    elif sentence[-1] == "1":
        label = [0,1,0]
    elif sentence[-1] == "2":
        label = [0,0,1]
    else:
        print("wrong")
    Y_test.append(label)
    # print("sum:%s" % vector)
    # print("divide:%s" % vector)
    # prediction = clf.predict(vector)
    # # print(prediction[0])
    # # print(sentence[-1])
    # if prediction[0] == sentence[-1]:
    #     right_count += 1

X_test.append(X_test[0])
X_test.append(X_test[0])
X_test.append(X_test[0])
X_test.append(X_test[0])
Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
X_test = sequence.pad_sequences(X_test, maxlen=552)

batch_size = 32
epochs = 6
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size, max_sentence_length, 100)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
# , validation_data=(X_val, Y_val)

model.save('lstm.h5')