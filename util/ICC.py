import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体来支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False
def three_parameter_logistic_model(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

# 设置参数值
a = 1  # 区分度
b = 0  # 难度
c = 0.25  # 猜测

# 设置能力水平的范围
theta = np.linspace(-4, 4, 100)

# 计算特征曲线
p_theta = three_parameter_logistic_model(theta, a, b, c)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(theta, p_theta, label='3PL Model')
plt.xlabel('学员能力 (θ)')
plt.ylabel('正确作答概率')
plt.title('项目特征曲线')
plt.grid(False)
plt.legend()
plt.savefig('ThreeParameterLogisticModel.png')
plt.show()
