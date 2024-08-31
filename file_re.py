import re
import os

# Aaron_Eckhart_0001.jpg 转换成 100-0.jpg
start = 99
num = 0
path = "./test"
test_name = "AaronEckhart"
for filename in os.listdir(path):
    last_name,turn_num  = filename.split("_0")
    print("last_name,test_name:",last_name,test_name)

    if last_name == test_name:
        num += 1
        save_name = str(start) + "-" + str(num) + ".jpg"
        os.rename(os.path.join(path,filename),os.path.join(path,save_name))

    else:
        num = 0
        start+=1
        test_name = last_name
        save_name = str(start) + "-" + str(num) + ".jpg"
        os.rename(os.path.join(path,filename),os.path.join(path,save_name))