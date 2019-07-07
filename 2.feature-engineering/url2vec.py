import gensim, logging

f = open("FinalDataWithoutLabel.txt", "r")
lines = f.readlines()
sentences = []
for line in lines:
    # print(line)
    line = line[:-1]
    sentence = line.split("\t")
    sentences.append(sentence)
print(len(lines))
print(len(sentences))

model = gensim.models.Word2Vec(sentences, min_count=1, size=100)
model.save("model")
print(model.wv['drivers'])