import numpy as np

from scipy.integrate import quad
import itertools
import sympy



# 被积函数
f = lambda x: -x**3*np.log(x) / ((x**2+1)*(x**3+1)) if x != 0 else 0
# 上下界
x_1 = 0
x_2 = np.inf

# 形式猜测
pi = np.pi # 圆周率
gamma = sympy.EulerGamma.n(20) # Euler常数
e = np.e # 自然常数
a = lambda x: -x[0]*pi**2/x[1]
ranges = [100,500]
#



'''
dx = 0.001
xM = 1000
s = 0


y_temp = 0
for x in np.arange(dx,xM,dx):
    y_new = f(x)
    s += (y_new + y_temp) * dx / 2
    y_temp = y_new
    
upper = - x**(-1) * (np.log(x) + 1) #x**(-4)/4 * np.log(x) + x**(-4)/16
s += upper
'''

s, error = quad(f, x_1, x_2)

res = []
for i in itertools.product(*[range(1, r + 1) for r in ranges]):
    s_temp = a(i)
    dist = np.abs(s_temp - s)
    # print(dist)
    if dist <= error:
        res.append({'param':i,'dist':dist})
        print('find {} / {} dist: {}'.format(s_temp, s ,dist))

res.sort(key=lambda x: x['dist'])


'''
res = []
for i1 in range(1,100):
    for i2 in range(1,2):
        for i3 in range(1,500):
            s_temp = a([i1,i2,i3])
            dist = np.abs(s_temp - s)
            # print(dist)
            if dist <= error:
                res.append({'param':[i1,i2,i3],'dist':dist})
                print('find {} / {} dist: {}'.format(s_temp, s ,dist))


res.sort(key=lambda x: x['dist'])
'''

'''
我想得到-0.8409405237658123
最接近的嵌套形式,有什么算法可以寻找
(嵌套的层数10以内,自然数20以内)
'''