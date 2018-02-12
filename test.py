
import numpy as np

def test():
<<<<<<< HEAD
    a = [1,2,3]
    b = [[1,2,3],
         [1,2,3]]
    c = np.multiply(a,b)
    print(c)
    sizes = [2,2,1]
    """"""
    weights_input_hidden = [np.random.randn(sizes[1], sizes[0])]  # 输入层到隐层之间的连接权
    weights_hidden_output = [np.random.randn(sizes[2], sizes[1])]  # 隐层到输出层之间的连接权
    threshold_hidden = [np.random.randn(sizes[1], 1)]  # 隐层的阈值
    threshold_output = [np.random.randn(sizes[2], 1)]  # 输出层的阈值

    print(weights_input_hidden[0])
    print(threshold_hidden)
    for w,t in zip(weights_input_hidden, threshold_hidden):
        print(w)
        print(t)

def loadWatermelon():
    fs = open("西瓜数据集 3.0.csv")
    arrayOLines = fs.readlines()
    arrayOLines.pop(0)
    numberLines = len(arrayOLines)
    data = np.zeros((numberLines,2))
    label = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split(',')
        data[index,:] = listFromLine[6:8]
        label.append(listFromLine[-1])
        index += 1
    labels = []
    for elements in label:
        if elements == '是':
            labels.append(1)
        else:
            labels.append(0)
    return data,labels
loadWatermelon()
=======
    d = 1
    q = 3
    l = 2
    sizes = [d, q, l]
    rand1 = np.random.randn(2, 4)
    print(rand1)
    thresholds= [np.random.randn(y, 1) for y in sizes[1:]]
    print(thresholds)
>>>>>>> 7547b1ac580471d8ccf23dbbefd2d7b944d34d67

test()