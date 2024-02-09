import matplotlib.pyplot as plt
import numpy as np

# 定义参数
interBSDist = 100
areaSize = np.array([0, 0]) * interBSDist

# 计算六边形顶点坐标
bsPositions = np.vstack([
    [areaSize / 2],
    np.array(
        [[areaSize[0]/2 + interBSDist * np.cos(a + np.pi/6),
          areaSize[1]/2 + interBSDist * np.sin(a + np.pi/6)]
         for a in np.linspace(0, 2*np.pi, 7)[:-1]])
])
mr = interBSDist * 2
l = mr * np.sin(np.pi/3)
bsPositions = np.vstack([
    bsPositions,
    np.array([[areaSize[0]/2 + mr * np.cos(a + np.pi/6), areaSize[1]/2 + mr * np.sin(a + np.pi/6)] for a in np.linspace(0, 2*np.pi, 6, endpoint=False)]),
    np.array([[areaSize[0]/2 + l * np.cos(a), areaSize[1]/2 + l * np.sin(a)] for a in np.linspace(0, 2*np.pi, 6, endpoint=False)])
])

r = interBSDist * 3
bsPositions = np.vstack([
    bsPositions,
    np.array([[areaSize[0]/2 + r * np.cos(a + np.pi/6), areaSize[1]/2 + r * np.sin(a + np.pi/6)] for a in np.linspace(0, 2*np.pi, 6, endpoint=False)]),
    np.array([areaSize[0]/2 + r * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * np.sin(np.pi/6)]),
    np.array([areaSize[0]/2 + r * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * np.sin(np.pi/6)]),
    np.array([areaSize[0]/2 + interBSDist * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * (2 + np.sin(np.pi/6))]),
    np.array([areaSize[0]/2 + mr * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * 2]),
    np.array([areaSize[0]/2 - interBSDist * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * (2 + np.sin(np.pi/6))]),
    np.array([areaSize[0]/2 - mr * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * 2]),
    np.array([areaSize[0]/2 - r * np.cos(np.pi/6), areaSize[1]/2 + interBSDist * np.sin(np.pi/6)]),
    np.array([areaSize[0]/2 - r * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * np.sin(np.pi/6)]),
    np.array([areaSize[0]/2 - interBSDist * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * (2 + np.sin(np.pi/6))]),
    np.array([areaSize[0]/2 - mr * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * 2]),
    np.array([areaSize[0]/2 + interBSDist * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * (2 + np.sin(np.pi/6))]),
    np.array([areaSize[0]/2 + mr * np.cos(np.pi/6), areaSize[1]/2 - interBSDist * 2])
])

print(len(bsPositions))

# 绘制六边形的角
plt.plot(bsPositions[:, 0], bsPositions[:, 1], marker='o', markersize=8, linestyle='', color='black')


# 设置坐标轴
plt.axis('equal')
plt.xlim(-500, 500)
plt.ylim(-500, 500)

# 显示图形
plt.show()