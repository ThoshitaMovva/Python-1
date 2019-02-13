# declared list 1
py = input("Enter the list of students who are in python class")
py_list = py.split(",")
# declared list 2
web = input("Enter list of students who are in web class")
w_list = web.split(",")
# created list for common students
common =[]
# created list for uncommon students
uncommon = []
for y in py_list:
    if y in w_list:
        common.append(y)
    else:
        uncommon.append(y)
for x in w_list:
    if x in py_list:
        pass
    else:
        uncommon.append(x)
# printing results
print("Students who are in python Class:",py_list)
print("Students who are in Web Class:",w-list)
print("common students:", common)
print("uncommon students:",uncommon)