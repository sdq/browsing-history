from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import gensim
import pickle

model = gensim.models.Word2Vec.load("model")

f = open('gradientboosting.pkl', 'rb')
clf = pickle.load(f)

researcher_sentences = []
supporter_sentences = []
purchaser_sentences = []
sentences = []

f = open('Test.txt', 'r')
lines = f.readlines()
lines_count = len(lines)
right_count = 0
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

    vector = np.zeros(100)
    urls = sentence[:-1]
    url_count = len(urls)
    if url_count == 0:
        continue
    # print("url_count:%s" % url_count)
    for url in urls:
        vector = np.add(vector, model[url])
    # print("sum:%s" % vector)
    vector = np.true_divide(vector, url_count)
    # print("divide:%s" % vector)
    prediction = clf.predict(vector)
    # print(prediction[0])
    # print(sentence[-1])
    if prediction[0] == sentence[-1]:
        right_count += 1

recall_purchaser_count = 0
precision_purchaser_count = 0
right_precision_purchaser_count = 0
recall_supporter_count = 0
precision_supporter_count = 0
right_precision_supporter_count = 0
recall_researcher_count = 0
precision_researcher_count = 0
right_precision_researcher_count = 0
for sentence in sentences:

    vector = np.zeros(100)
    urls = sentence[:-1]
    url_count = len(urls)
    if url_count == 0:
        continue
    for url in urls:
        vector = np.add(vector, model[url])
    vector = np.true_divide(vector, url_count)
    prediction = clf.predict(vector)
    # print(prediction[0])
    # print(sentence[-1])

    if prediction[0] == "0":
        precision_researcher_count += 1
        if sentence[-1] == "0":
            right_precision_researcher_count += 1

    if prediction[0] == "1":
        precision_supporter_count += 1
        if sentence[-1] == "1":
            right_precision_supporter_count += 1

    if prediction[0] == "2":
        precision_purchaser_count += 1
        if sentence[-1] == "2":
            right_precision_purchaser_count += 1
    

print("general:%s/%s" % (right_count,lines_count))

print("purchaser######")
print("recall:%s/%s" % (right_precision_purchaser_count,len(purchaser_sentences)))
print("precision:%s/%s" % (right_precision_purchaser_count,precision_purchaser_count))

print("supporter######")
print("recall:%s/%s" % (right_precision_supporter_count,len(supporter_sentences)))
print("precision:%s/%s" % (right_precision_supporter_count,precision_supporter_count))

print("researcher######")
print("recall:%s/%s" % (right_precision_researcher_count,len(researcher_sentences)))
print("precision:%s/%s" % (right_precision_researcher_count,precision_researcher_count))

# general:928/1256=73.8853503%
# purchaser######
# recall:75/228=32.894736842%
# precision:75/147=51.020408163%
# supporter######
# recall:579/657=88.127853881%
# precision:579/655=88.396946565%
# researcher######
# recall:274/371=73.854447439%
# precision:274/442=61.990950226%