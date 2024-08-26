# miniGPT2
从零实现一个迷你版本的GPT2

训练曲线

![image](https://github.com/user-attachments/assets/16ff2f30-4c15-4b69-92d6-656efd1fc5cf)

相关组件请看我的transformer的代码仓库, 这里使用了transformer的decoder部分, 我使用了8层decoder进行累加, 实现了一个小型的GPT. 数据集较小所以有轻微过拟合, 但是损失函数总体向下. 

预测效果

![image](https://github.com/user-attachments/assets/f5aa83ff-a2a9-45ab-af3d-7e6b4f077d6c)
