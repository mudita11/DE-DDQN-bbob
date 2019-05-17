import re

opt = []
with open('alloptimas_bbob', 'r') as f:
    for line in f:
        #a = line.split(' '); print(a)
        b = line.rstrip()
        opt.append(b)

tr = []
with open('training_set', 'r') as f:
    for line in f:
        #a = line.split(' '); print(a)
        b = line.rstrip()
        tr.append(b)
#print(tr)

count = 0
file = open('optima_for_training', 'a')
opt_tr = []
for i in range(len(tr)):
    opt_tr.append(opt[int(tr[i])])
    file.write(str(opt[int(tr[i])])+'\n')
    count +=1
print(count)
#print(opt_tr)
