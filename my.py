# players = ['charles','martina','michael','florence','eli']
#
# print("Here are the first three players on my team:")
# for player in players[0:3]:
#     print(player.title())

#foods.py
# my_foods = ['稀饭','米饭','拉条子']
# friend_foods = my_foods[:]
#
# print(my_foods)
# print(friend_foods)
#
# my_foods.append("麻什")
# friend_foods.append("凉皮")
#
# print(my_foods)
# print(friend_foods)
#
# s_foods = my_foods
# my_foods.append("ice cream")
# print(my_foods)
# print(s_foods)

#amusement_park.py
# age = 12
# if age < 4:
#     print("Your admission cost is $0.")
# elif age < 18:
#     print("Your admission cost is $5.")
# else:
#     print("Your admission cost is $10.")

# available_toppings = ['mushroom','olibers','green peppers',
#                       'pepperoni','pineapple','extra cheese']
#
# requested_toppings = ['mushroom','french fries','extra cheese']
#
# for requested_topping in requested_toppings:
#     if requested_topping in available_toppings :
#         print("Adding " + requested_topping + ".")
#     else :
#         print("Sorry,we don't have " + requested_topping + ".")
# print("\nFinished making your pizza!")

#dimensions.py元组


#第六章 字典
#alien.py
# alien_O = {'color' : 'green','points' : '5'}
#
# print(alien_O['color'])
# print(alien_O['points'])
#
# alien_O['x_position'] = 0
#
# print(alien_O)
# del alien_O['x_position']
#
# print(alien_O)
# alien_O = {'color' : 'green','points' : '5'}
# alien_1 = {'color' : 'yellow','points' : '10'}
# alien_2 = {'color' : 'red','points' : '15'}
#
# aliens = [alien_O,alien_1,alien_2]
#
# for alien in aliens:
#     print(alien)

#many_users.py

# users = {
#     'aeinstein' : {
#         'first' : 'albert',
#         'last' :  'einstein',
#         'location': 'princeton',
#     },
#
#     'murie' : {
#         'first' : 'marie',
#         'last' :  'curie',
#         'location': 'paris',
#     },
# }

# for username,user_info in users.items():
#     print("\nUsername: " + username)
#     full_name = user_info['first'] + " " + user_info['last']
#     location = user_info['location']
#
#     print("\tFull name: " + full_name.title())
#     print("\tLocation: " + location.title())

#第七章 用户输入和while循环
#parrot.py

# message = input("Tell me something ,and I will repeat it back to you :")
# print("用户输入的信息: " + message)

# age = input("How old are you? ")
#counting.py
# current_number = 1
# while current_number <= 5:
#     print(current_number)
#     current_number += 1

#confirmed_users.py
# unconfirmed_users = ['alice','brian','candace']
# confirmed_users = []
#
# while unconfirmed_users:
#     current_user = unconfirmed_users.pop()
#
#     print("\nVerifying user: " + current_user.title())
#     confirmed_users.append(current_user)
#
# print("\nThe following usres have been confirmed:")
# for confirmed_user in confirmed_users:
#     print(confirmed_user.title())

#第八章 函数

#greety.py

# def greet_user():
#     """显示简单的问候语"""
#     print("Hello!")
#
# #调用函数
# greet_user()

# for value in range(5):
#     print(value)

# def greet_user(username):
#     """显示简单的问候语"""
#     print("Hi, " + username.title() + '~')
#
# greet_user('tom')

# def describe_pet(pet_name,animal_type = 'dog'):
#     print("\nI have a " + animal_type.title() + ".")
#     print("My " + animal_type + "'s name is " + pet_name.title() + ".")
#
# describe_pet(pet_name='willie')

#formatted_name
# def get_formatted_name(first_name, last_name):
#     full_name = first_name + ' ' + last_name
#     return full_name.title()
# musician = get_formatted_name('jimi', 'hendrix')
# print(musician)

#第九章 类
#dog.py

# class Dog():
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def sit(self):
#         print(self.name.title() + " is now sitting.")
#
#     def roll_over(self):
#         print(self.name.title() + " rolled over!")
# my_dog =Dog('willie',6)
# print("My dog's name is " + my_dog.name.title() + ".")
# print("My dog is " + str(my_dog.age) + " years old.")

#第十五章 绘制简单的折线图
#matplotlib 画廊
# import matplotlib.pyplot as plt
#
# input_values = [1,2,3,4,5]
# squares = [1,4,9,16,25]
# plt.plot(input_values, squares, linewidth=5)
# #设置图表标题，并给坐标轴加上标签
# plt.title("Square Numbers", fontsize=24)
# plt.xlabel("Value",fontsize=14)
# plt.ylabel("Square of Value",fontsize=14)
#
# #设置刻度标记的大小
# plt.tick_params(axis='both', labelsize=14)
# plt.show()

#使用scatter()绘制散点图并设置其样式
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

#随机漫步

# from random import choice
#
# class RandomWalk():
#     """一个生成随机漫步数据的类"""
#     def __init__(self, num_points=5000):
#         """初始化随机漫步的属性"""
#         self.num_points = num_points
#
#         #虽有随机漫步都始于[0,0]
#         self.x_values = [0]
#         self.y_values = [0]
#
#     def fill_walk(self):
#         """计算随机漫步包含的所有点"""
#
#         #不断漫步，直到列表达到指定的长度
#         while len(self.x_values) < self.num_points:
#
#             # 决定前进方向以及沿这个方向前进的距离
#             x_direction = choice([1,-1])
#             x_distance = choice([0,1,2,3,4])
#             x_step = x_direction * x_distance
#
#             y_direction = choice([1,-1])
#             y_distance = choice([0,1,2,3,4])
#             y_step = y_direction * y_distance
#
#             #拒绝原地踏步
#             if x_step == 0 and y_step == 0:
#                 continue
#
#             #让下一个点的x和y值
#             next_x = self.x_values[-1] + x_step
#             next_y = self.y_values[-1] + y_step
#
#             self.x_values.append(next_x)
#             self.y_values.append(next_y)
#
# rw = RandomWalk(50000)#增加点数
# rw.fill_walk()
# point_numbers = list(range(rw.num_points))
#
# plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues, edgecolors='none', s=15)
#
# #突出起点和终点
# plt.scatter(0, 0, c='green', edgecolors='none', s=100)
# plt.scatter(rw.x_values[-1], rw.x_values[-1], c ='red', edgecolors='none',s=100)
#
# #隐藏坐标轴
#
# # plt.axes().get_xaxis().set_visible(False)
# # plt.axes().get_yaxis().set_visible(False)
# plt.show()

#使用Pygal模拟掷骰子

#第16章 下载数据

from numpy import *
import csv
#filename = '西瓜数据集 3.0.csv'
filename = 'ccf_online_stage1_train.csv'

# with open(filename) as f:
#     reader = csv.reader(f)
#     header_row = next(reader)
#     # print(header_row)
#     #
#     # for index, column_header in enumerate(header_row):
#     #     print(index,column_header)
#
#     highs = []
#     for row in reader:
#         highs.append(row)
#
#     print(len(array(highs)))

#打开超大的csv,控制按行读取
#天池大数据竞赛小试牛刀，路还长着呢，吾将上下而求索，一人。
with open('ccf_online_stage1_train.csv', 'r') as fin:
    block = []
    for line in fin:
        block.append(line)
            # if len(block) <= 1000000:
            #     fout.write(','.join(line.split(' ')))
            # else:
            #     break
    print(len(block))


