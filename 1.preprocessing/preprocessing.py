from random import shuffle
import re

f = open('FinalData.txt', 'r')
lines = f.readlines()
shuffle(lines)
print(len(lines))
output_file = open("NewFinalData.txt", "w")
replace_func = lambda x:x.replace("http://","").replace("https://","")
split_func = lambda x:x.split("/")
re_func = lambda x:re.findall(r"[\w']+", x)
pop_func = lambda x:x[1:]
new_lines = []
for line in lines:
    new_line = ""
    line = line[:-1]
    print(line[-2:])
    # lines = lines.replace("http://","")
    # lines = lines.replace("https://","")
    sentence = line.split("\t")
    sentence = list(map(replace_func, sentence))
    words = list(map(split_func, sentence))
    pop_words = list(map(pop_func, words[:-1]))
    pop_words.append(words[-1])
    new_words = []
    for word in pop_words:
        new_words.extend(word)
    new_words = list(map(re_func, new_words))
    final_words = []
    for word in new_words:
        final_words.extend(word)
    for word in final_words:
        new_line += "%s\t" % word
    new_lines.append(new_line)

for line in new_lines:
    line = line[:-1]
    output_file.write("%s\n" % line)
output_file.close()
