import yaml
import os
import numpy as np
import pandas as pd
import torch

# 获取当前脚本的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))
print(base_path)
# 构建 YAML 文件的路径
function_file_path = os.path.join(base_path,  'config', 'function.yaml')
# TODO 修改函数名称生成不一样的数据
func = 'function6'


with open(function_file_path, 'r') as file:
    data = yaml.safe_load(file)
    function_name = data[func]['name']
    function = data[func]['expression']
    variables_name = data[func]['variables']
    variables_num = len(variables_name)
    train_num = data[func]['train']
    val_num = data[func]['val']
    test_num = data[func]['test']
    # variables={variable: None for variable in variables}

np.random.seed(42)
train_data_variables = np.random.uniform(0, 10, (train_num, variables_num))
np.random.seed(43)
val_data_variables = np.random.uniform(0, 10, (val_num, variables_num))
np.random.seed(44)
test_data_variables = np.random.uniform(0, 10, (test_num, variables_num))

# train_data_variables = np.linspace(0, 100, train_num * variables_num)
# 重新形状化为(train_num, variables_num)
train_data_variables = train_data_variables.reshape((train_num, variables_num))

# val_data_variables = np.linspace(0, 100, val_num * variables_num)
# 重新形状化为(train_num, variables_num)
val_data_variables = val_data_variables.reshape((val_num, variables_num))

# test_data_variables = np.linspace(0, 100, test_num * variables_num)
test_data_variables = test_data_variables.reshape((val_num, variables_num))


    
def generate(mode='val',func=function,train_data_variables=train_data_variables):
    func = func.replace("exp", "np.exp").replace("sin", "np.sin").replace("cos", "np.cos").replace("log", "np.log")
    variables = []
    variables_T = []
    if mode=='train':
        save_path = os.path.join(base_path,  'fitting_data', f'{function_name}_train.csv')
        variables = train_data_variables
        variables_T = train_data_variables.transpose()
    if mode=='val':
        save_path = os.path.join(base_path,  'fitting_data', f'{function_name}_val.csv')
        variables = val_data_variables
        variables_T = val_data_variables.transpose()
    if mode=='test':
        save_path = os.path.join(base_path,  'fitting_data', f'{function_name}_test.csv')
        variables = test_data_variables
        variables_T = test_data_variables.transpose()
    x=variables_T[0]
    y=variables_T[1]
    z=variables_T[2]
    w=variables_T[3]
    e=variables_T[4]
    r=variables_T[5]
    t=variables_T[6]
    g=variables_T[7]

    # TODO 需要根据函数变量名称手动修改local_env
    local_env = {'x': x, 'y': y, 'z': z, 'w': w, 'e': e,'r': r,'t':t,'g':g, 'np': np}
    # local_env = {'x': x, 'y': y,'np': np}
    # local_env = {'x': x,'np': np}
    result = eval(func, {}, local_env)
    result = result.reshape(-1, 1)  # -1 在这里意味着自动计算行数
    print(result)
    # 沿着水平方向拼接两个数组
    combined_array = np.concatenate((variables, result), axis=1)
    print(combined_array.shape)
    # Todo 修改变量
    df = pd.DataFrame(combined_array, columns=['x','y','z','w','e','r','t','g', 'result'])
    # 保存 DataFrame 到 CSV 文件
    df.to_csv(save_path, index=False)
    
modes = ['train', 'val', 'test']
for mode in modes:
    generate(mode=mode)








