#-*- coding:utf-8 -*-
f = open('/data/face/label.txt')
r = f.readlines()
f2 = open('/data/face/msceleb1m_images.txt','w')
l = len(r)
for i in range(l):
    w = r[i].split(' ')[0]
    s = w.split('/')
    s[1] = 'data'
    content = '/'.join(s)
    f2.write(content+'\n')
    if i%10000 == 0:
        print('Done txt:%.2f'%(i/l))
f2.close()