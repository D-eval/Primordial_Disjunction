import numpy as np
import sympy
import random

pi = np.pi # 圆周率
gamma = sympy.EulerGamma.n(20) # Euler常数
e = np.e # 自然常数


def set_seed(seed: int = 42):
    """设置全套随机种子"""
    # 设置 Python 内置随机种子
    random.seed(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)


set_seed(42)



f0 = lambda x: x[0] / x[1]
f1 = lambda x: pi**x[0] * x[1] / x[2]
f2 = lambda x: np.exp(x[0]/x[1]) * x[2] / x[3]
f3 = lambda x: x[0]/x[1] * np.log(x[2])
f4 = lambda x: x[0]/x[1] * np.sqrt(x[2])

idx_to_f_ranges = {
    0:(f0, [100,100]),
    1:(f1, [5,10,10]),
    2:(f2, [3,3,10,10]),
    3:(f3, [10,10,100]),
    4:(f4, [10,10,100])
}

class Form:
    def __init__(self,args):
        f_idx, x = args
        self.f_idx = f_idx
        self.x = x
    def value(self):
        f, _ = idx_to_f_ranges[self.f_idx]
        x = self.x
        return f(x)
    def __repr__(self):
        f_idx = self.f_idx
        x = self.x
        x_str = ','.join([str(i) for i in x])
        return 'f{}([{}])'.format(f_idx,x_str)


def sample_I():
    form_idx = np.random.randint(0,4)
    f, ranges = idx_to_f_ranges[form_idx]
    x = [random.randint(1, limit-1) for limit in ranges]
    form = Form((form_idx,x))
    # 错误条件
    if form_idx == 3:
        if [2] == 1:
            return sample_I()
    if form_idx == 4:
        if (np.sqrt(x[2]) % 1) == 0:
            return sample_I()
    return form


def is_simplest(x1:Form,x2:Form):
    if x1.f_idx != x2.f_idx:
        return True
    idx = x1.f_idx
    # 返回“可使用”的必要条件
    if idx == 0:
        return x1.x[1] != x2.x[1]
    elif idx == 1:
        return x1.x[0] != x2.x[0]
    elif idx == 2:
        return x1.x[0]/x1.x[1] != x2.x[0]/x2.x[1]
    elif idx == 3:
        return x1.x[2] != x2.x[2]
    elif idx == 4:
        return x1.x[2] != x2.x[2]
    else:
        raise ValueError('形式错误')


def sample_I_add_I():
    x1 = sample_I()
    x2 = sample_I()
    if is_simplest(x1,x2):
        return x1,x2
    else:
        return sample_I_add_I()


def sample_data():
    form1,form2 = sample_I_add_I()
    x1,x2 = form1.value(),form2.value()
    x = x1 + x2
    scale = 1
    while x / scale >= 1:
        scale *= 2
    x1 /= scale
    x2 /= scale
    return {'data':(x1,x2),
            'scale':scale,
            'form':(form1,form2)}


class ValidSet():
    def __init__(self, valid_num=100):
        self.num = valid_num
        data = []
        for i in range(valid_num):
            data_temp = sample_data()
            data.append(data_temp)
        self.data = data
    def __len__(self):
        return self.num
    def __getitem__(self,i):
        return self.data[i]


validset = ValidSet()
