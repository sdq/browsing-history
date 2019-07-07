from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.models import load_model
import numpy as np
import gensim
import pickle

model = gensim.models.Word2Vec.load("model")
lstm_model = load_model('lstm.h5')

researcher_sentences = []
supporter_sentences = []
purchaser_sentences = []

sentences = []

f = open('Test.txt', 'r')
lines = f.readlines()
lines_count = len(lines)
right_count = 0
wrong_count = 0
X = []
Y = []
Y_test = []
for line in lines:
    line = line[:-1]
    sentence = line.split("\t")

    if sentence[-1] == "0":
        researcher_sentences.append(sentence)
    elif sentence[-1] == "1":
        supporter_sentences.append(sentence)
    elif sentence[-1] == "2":
        purchaser_sentences.append(sentence)

    sentences.append(sentence)

    vectors = []
    urls = sentence[:-1]
    url_count = len(urls)
    # print("url_count:%s" % url_count)
    if url_count == 0:
        continue
    for url in urls:
        vectors.append(model[url])
    X.append(vectors)

    label = -1
    if sentence[-1] == "0":
        label = 0
    elif sentence[-1] == "1":
        label = 1
    elif sentence[-1] == "2":
        label = 2
    else:
        print("wrong")
    Y.append(label)

    label1 = [0,0,0]
    if sentence[-1] == "0":
        label1 = [1,0,0]
    elif sentence[-1] == "1":
        label1 = [0,1,0]
    elif sentence[-1] == "2":
        label1 = [0,0,1]
    else:
        print("wrong")
    Y_test.append(label1)
    # print("sum:%s" % vector)
    # print("divide:%s" % vector)
    # prediction = clf.predict(vector)
    # # print(prediction[0])
    # # print(sentence[-1])
    # if prediction[0] == sentence[-1]:
    #     right_count += 1

Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
Y_test.append(Y_test[0])
X.append(X[0])
X.append(X[0])
X.append(X[0])
X.append(X[0])
X_test = sequence.pad_sequences(X, maxlen=552)
Px = lstm_model.predict(X_test)
Px = Px[:-4]

for i, p in enumerate(Px):
    p_index = np.argmax(p)
    label = Y[i]
    if p_index == label:
        right_count += 1
    else:
        wrong_count += 1
        print("wrong:%s,p:%s,l:%s" % (wrong_count,p_index,label))

recall_purchaser_count = 0
precision_purchaser_count = 0
right_precision_purchaser_count = 0
recall_supporter_count = 0
precision_supporter_count = 0
right_precision_supporter_count = 0
recall_researcher_count = 0
precision_researcher_count = 0
right_precision_researcher_count = 0
for i, y in enumerate(Y[:-4]):
    # print(prediction[0])
    # print(sentence[-1])

    p_index = np.argmax(Px[i])
    if p_index == 0:
        precision_researcher_count += 1
        if y == 0:
            right_precision_researcher_count += 1

    if p_index == 1:
        precision_supporter_count += 1
        if y == 1:
            right_precision_supporter_count += 1

    if p_index == 2:
        precision_purchaser_count += 1
        if y == 2:
            right_precision_purchaser_count += 1
    
score = lstm_model.evaluate(X_test, Y_test, batch_size=32)

print("score:%s" % score)

print("general:%s/%s" % (right_count,len(Px)))

print("purchaser######")
print("recall:%s/%s" % (right_precision_purchaser_count,len(purchaser_sentences)))
print("precision:%s/%s" % (right_precision_purchaser_count,precision_purchaser_count))

print("supporter######")
print("recall:%s/%s" % (right_precision_supporter_count,len(supporter_sentences)))
print("precision:%s/%s" % (right_precision_supporter_count,precision_supporter_count))

print("researcher######")
print("recall:%s/%s" % (right_precision_researcher_count,len(researcher_sentences)))
print("precision:%s/%s" % (right_precision_researcher_count,precision_researcher_count))


# general:891/1256=73.7261146%
# purchaser######
# recall:73/228=31.578947368%
# precision:73/179=51.428571429%
# supporter######
# recall:577/657=89.193302892%
# precision:577/659=88.922610015%
# researcher######
# recall:241/371=74.393530997%
# precision:241/406=62.02247191%