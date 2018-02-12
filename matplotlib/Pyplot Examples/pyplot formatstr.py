import matplotlib.pyplot as plt

# plt.scatter(2,4, s=200)#s=200加重点的大小 着墨
# #设置图表标题并给坐标轴加上标签
# plt.title("Squares Numbers", fontsize=24)
# plt.xlabel("Value", fontsize=14)
# plt.ylabel("Square of Value", fontsize=14)
# plt.show()

# x_values = list(range(1,1001))
# y_values = [x**2 for x in x_values]

#plt.scatter(x_values, y_values, c=(0, 0, 0.8), edgecolors='none', s=40)
# plt.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Blues, edgecolors='none', s=40)
#设置图表标题并给坐标轴指定标签
# plt.axis([0,1100,0,1100000])
# plt.show()
#plt.savefig('squares_plog.png', bbox_inches = 'tight') 自动保存图表
x = [0,1,2,3,4]
y = [0,1,4,9,16]
plt.plot(x,y,'ro')
scatters = [0,6,0,20]
plt.axis(scatters)
plt.show()