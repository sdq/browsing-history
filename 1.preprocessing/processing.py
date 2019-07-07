from random import shuffle

f = open('NewFinalData.txt', 'r')
lines = f.readlines()
shuffle(lines)
print(len(lines))
output_file = open("FinalDataWithoutLabel.txt", "w")
for line in lines:
    line = line[:-3]
    output_file.write("%s\n" % line)
output_file.close()

train_file = open("Train.txt", "w")
test_file = open("Test.txt", "w")

lines_0 = []
lines_1 = []
lines_2 = []
count_0 = 0
count_1 = 0
count_2 = 0

for line in lines:
    print(line[-2])
    if line[-2] == "0":
        count_0 += 1
        lines_0.append(line)
    elif line[-2] == "1":
        count_1 += 1
        lines_1.append(line)
    else:
        count_2 += 1
        lines_2.append(line)
print([count_0, count_1, count_2])

train_lines = []
test_lines = []

test_count_0 = len(lines_0)/10
train_lines_0 = lines_0[:-test_count_0]
test_lines_0 = lines_0[-test_count_0:]
print(len(train_lines_0))
print(len(test_lines_0))

test_count_1 = len(lines_1)/10
train_lines_1 = lines_1[:-test_count_1]
test_lines_1 = lines_1[-test_count_1:]
print(len(train_lines_1))
print(len(test_lines_1))

test_count_2 = len(lines_2)/10
train_lines_2 = lines_2[:-test_count_2]
test_lines_2 = lines_2[-test_count_2:]
print(len(train_lines_2))
print(len(test_lines_2))

train_lines = train_lines_0 + train_lines_1 + train_lines_2
test_lines = test_lines_0 + test_lines_1 + test_lines_2
shuffle(train_lines)
shuffle(test_lines)

print("train:%s, test:%s" % (len(train_lines),len(test_lines)))

for train_line in train_lines:
    train_file.write("%s" % train_line)
for test_line in test_lines:
    test_file.write("%s" % test_line)

train_file.close()
test_file.close()
