'''
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
print(t)
print(s)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)
plt.savefig("test.png")
plt.show()
'''

import re
import numpy as np


def precision_recall_k(filename):
    with open(filename, "rb") as f:
        data1 = f.read().split('\n')
    d1 = []
    d2 = []
    count = 0
    for d in data1:
        count += 1
        if count>10:
            break

        split_data = d.split("\t")
        d1.append(split_data[0])


    matrix = []
    for f1 in range(0,len(d1)):
        matrix.append([])
        for f2 in range(0,len(d1)):
            tup = (d1[f1],d1[f2],0.0,0.0)
            matrix[f1].append(tup)

    print matrix
precision_recall_k("pairs.txt")

a = [(0,0),(0,0)(0,0)(0,0)(0,0)(0,0)(0,0)(0,0)]
for i in a:
    a[0] = 1
    a[1] = 9
print a



if True:
    exit()
x = None
def tester():
    a=[1,20]
    b=[5,100]
    return a,b
a,b = tester()
print a
print b
if True:
    exit()
def convertAlphatoNum(input):
    x = []
    x.append(2)
    x.append(4)
    non_decimal = re.compile(r'[^\d.\s\w]+')
    return non_decimal.sub('', input)

def test():
    return 1,2
a,b = test()
print a
print b
if True:
    exit()

print float(convertAlphatoNum("5"))
print x
v = "55"
print v.lower()

a = [1, 2, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2]

print np.argmax(np.bincount(a))

from langdetect import detect

print detect("I am going home")

l1 = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
l2 = [7, 15, 36, 39, 40, 41]

print np.percentile(l1, 25)
print np.percentile(l1, 50)
print np.percentile(l1, 75)

print np.percentile(l2, 25)
print np.percentile(l2, 50)
print np.percentile(l2, 75)
if True:
    exit()
