import os
import matplotlib.pyplot as plt
from Star_measure import image_process as ip
import shutil

path = ".\\Fit_Images\\Training\\"
x = os.listdir(path)
print(x,len(x))

## Only works on domain computers
obbo_path = "\\obbo\\d$\\Images\\allskyeye\\images_in\\"
obbo_path = "//obbo/d$/Images/allskyeye/images_in/"
##

if len(x) != 999:
    try:
        y = os.listdir(obbo_path)
    except FileNotFoundError:
        print("Obbo computer not avaliable.")
    else:
        y_list = []
        for i in range(1000 -len(x)):
            print(i)
            try:
                element = y[round(30*i)]
                shutil.copy(obbo_path + element, path+element)
            except IndexError:
                print("Error: IndexError")
                continue
            except FileNotFoundError:
                print("Error: FileNotFoundError")

path = ".\\Fit_Images\\Training\\"
x = os.listdir(path)
print(x,len(x))
for i in range(len(x)):
    print(i)
    if "___" not in x[i]:
        print(x[i],path)
        data = ip([str(x[i])],path)
        data.show_img()
        label = str(input("Enter estimated cloud cover :"))
        name = x[i]
        os.rename(path+str(name),path+name[len(name)-4]+"___"+label+".fit")
        print("Renamed")
    else:
        print("Already done.")