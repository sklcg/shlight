import os

test_set = list(range(1,32))
train_set = [_ for _ in range(1, 401) if _ not in test_set]

def getSHMean(shfile):
    with open(shfile) as f:
        first_line = f.readline()
    x = sum(map(float,first_line.split()))
    return x / 3

with open('train.txt','w') as f:
    for i in train_set:
        for j in range(512):
            rng = getSHMean('../sh/%04d/%03d.txt'%(i,j))
            # filter out images under/over exposure
            if 0.3 < rng < 4:
                f.write('../data/im/%04d/%03d_1.jpg '%(i,j))
                f.write('../data/im/%04d/%03d_0.jpg '%(i,j))
                f.write('../data/sh/%04d/%03d.txt\n'%(i,j))
print('train.txt: done')

with open('test.txt','w') as f:
    for i in test_set:
        for j in range(512):
            rng = getSHMean('../sh/%04d/%03d.txt'%(i,j))
            # filter out images under/over exposure
            if 0.3 < rng < 4:
                f.write('../data/im/%04d/%03d_1.jpg '%(i,j))
                f.write('../data/im/%04d/%03d_0.jpg '%(i,j))
                f.write('../data/sh/%04d/%03d.txt\n'%(i,j))
print('test.txt: done')
