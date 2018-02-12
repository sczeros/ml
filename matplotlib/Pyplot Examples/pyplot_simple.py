# matplotlib类库学习第一次

import matplotlib.pyplot as plt
x = [0,1,2,3,4]
y = [0,1,2,3,4]
#fontsize Valid font size are xx-small, x-small, small, medium,
# large, x-large, xx-large, larger, smaller, None
#color
plt.title("pyplot simple",fontsize=22)
plt.xlabel("x",fontsize=18, color="green")
plt.ylabel("y")
plt.plot(x, y,linewidth=5)
plt.show()