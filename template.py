forward_template = [
    # 普通加法
    "add leaf leaf",
    
    # 普通乘法
    "mul leaf leaf",
    
    # 普通减法
    "add neg leaf leaf",
    "add leaf neg leaf",
    
    # 普通除法
    "mul leaf inv leaf",
    "mul inv leaf leaf",
    
    # 混合运算
    # "mul leaf add leaf leaf",
    # "mul add leaf leaf leaf",
    # "add mul leaf leaf mul leaf leaf",
    
]

inverse_template = [
    # 2 数之和
    "add sqrt leaf sqrt leaf",
    "add sqrt leaf leaf",
    "add sqrt leaf inv leaf",
    
    # 3 数字之和
    "add sqrt leaf add sqrt leaf sqrt leaf",
    
    # 2 数加权和
    "add mul leaf sqrt leaf mul leaf sqrt leaf",
]
