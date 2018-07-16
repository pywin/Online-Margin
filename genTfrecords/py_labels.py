#-*- coding:utf-8 -*-
f = open('/data/face/label.txt')
r = f.readlines()
f2 = open('/data/face/msceleb1m_labels.txt','w')
l = len(r)
for i in range(l):
    w = r[i].split(' ')[1]
    f2.write(w)
    if i%10000 == 0:
        print('Done txt:%.2f'%(i/l))
f2.close()