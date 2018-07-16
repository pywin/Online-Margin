#-*- coding:utf-8 -*-

# f2 = open('/data/face/msceleb1m_symbols.txt','w')
f = open('4.txt','w')
for i in range(40):
    f.write(str(i)+'\n')
    if i == 39:
        f.write(str(i))
f.close()