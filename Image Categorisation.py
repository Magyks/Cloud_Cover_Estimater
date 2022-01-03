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
time = float(input("Duration of image shown?  :"))
for i in range(len(x)):
    print(i)
    if "___" not in x[i]:
        print(x[i],path)
        data = ip([str(x[i])],path)
        data.show_img(time=time)
        label = str(int(input("Enter estimated cloud cover :")))
        name = x[i]
        new_name = path+name[:len(name)-4]+"___"+label+".fit"
        os.rename(path+str(name),new_name)
        print("Renamed from :",path+str(name),"to :",new_name)
    else:
        print("Already done.")