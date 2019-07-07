from sklearn.ensemble import GradientBoostingClassifier
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
    # print("url_count:%s" % url_count)
    if url_count == 0:
        continue
    for url in urls:
        vector = np.add(vector, model[url])
    # print("sum:%s" % vector)
    vector = np.true_divide(vector, url_count)
    # print("divide:%s" % vector)
    X.append(vector)
    if X == None:
        print("none")
    Y.append(sentence[-1])

print(Y)
print(X[-1])

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf = clf.fit(X, Y)

with open('gradientboosting.pkl', 'wb') as f:
    pickle.dump(clf, f)


