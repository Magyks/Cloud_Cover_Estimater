import os
import matplotlib.pyplot as plt
from Star_measure import image_process as ip
import shutil

path = "C:\\Users\\alexh\\Documents\\Scripts\\Fit_Images\\Training\\"
x = os.listdir(path)
print(x,len(x))

obbo_path = "\\obbo\\d$\\Images\\allskyeye\\images_in\\"
obbo_path = "//obbo/d$/Images/allskyeye/images_in/"
if len(x) != 999:
    y = os.listdir(obbo_path)
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

path = "C:\\Users\\alexh\\Documents\\Scripts\\Fit_Images\\Training\\"
x = os.listdir(path)
print(x,len(x))
for i in range(len(x)):
    print(i)
    if "___" not in x[i]:
        print(x[i],path)
        data = ip([str(x[i])],path)
        data.show_img()
        label = str(input("Enter estimated cloud cover :"))
        os.rename(path+str(x[i]),path+x[i]+"___"+label+".fit")
        print("Renamed")
    else:
        print("Already done.")