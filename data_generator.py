import yaml
import os
import numpy as np
import pandas as pd

# 获取当前脚本的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))
print(base_path)
# 构建 YAML 文件的路径
function_file_path = os.path.join(base_path,  'config', 'function.yaml')


with open(function_file_path, 'r') as file:
    data = yaml.safe_load(file)
    function_name = data['function1']['name']
    function = data['function1']['expression']
    variables_name = data['function1']['variables']
    variables_num = len(variables_name)
    train_num = data['function1']['train']
    val_num = data['function1']['val']
    test_num = data['function1']['test']
    # variables={variable: None for variable in variables}

np.random.seed(42)
train_data_variables = np.random.uniform(0, 1000, (train_num, variables_num))
np.random.seed(43)
val_data_variables = np.random.uniform(0, 1000, (val_num, variables_num))
np.random.seed(44)
test_data_variables = np.random.uniform(0, 1000, (test_num, variables_num))

    
def generate(mode='val',func=function,train_data_variables=train_data_variables):
    func = func.replace("exp", "np.exp").replace("sin", "np.sin").replace("cos", "np.cos").replace("log", "np.log")
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
    local_env = {'x': x, 'y': y, 'z': z, 'w': w, 'np': np}
    result = eval(func, {}, local_env)
    result = result.reshape(-1, 1)  # -1 在这里意味着自动计算行数
    print(result)
    # 沿着水平方向拼接两个数组
    combined_array = np.concatenate((variables, result), axis=1)
    print(combined_array.shape)
    df = pd.DataFrame(combined_array, columns=['x', 'y', 'z', 'w', 'result'])
    # 保存 DataFrame 到 CSV 文件
    df.to_csv(save_path, index=False)
    

generate()








