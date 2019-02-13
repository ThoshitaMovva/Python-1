input = [('John', ('Physics', 80)), ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)),
         ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
dict = {}
for i in input:
    tuple = i
    if tuple[0] in dict.keys():
        dict[tuple[0]].append(tuple[1])
    else:
        dict[tuple[0]] = [tuple[1]]

for x,y in dict.items():
    print(x, sorted(y))
