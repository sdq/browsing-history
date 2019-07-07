from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import gensim
import pickle

model = gensim.models.Word2Vec.load("model")

f = open('Train.txt', 'r')
lines = f.readlines()
X = []
Y = []
for line in lines:
    line = line[:-1]
    sentence = line.split("\t")
    vector = np.zeros(100)
    urls = sentence[:-1]
    url_count = len(urls)
    print(url_count)
    if url_count == 0:
        continue
    # print("url_count:%s" % url_count)
    for url in urls:
        vector = np.add(vector, model[url])
    # print("sum:%s" % vector)
    vector = np.true_divide(vector, url_count)
    # print("divide:%s" % vector)
    X.append(vector)
    Y.append(sentence[-1])

print(Y)
print(X[-1])

clf = AdaBoostClassifier(n_estimators=100)
clf = clf.fit(X, Y)

with open('adaboost.pkl', 'wb') as f:
    pickle.dump(clf, f)


