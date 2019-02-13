input_string = input("enter the string ")
crnt_longest = ""
lngt_string = ""
sub_string = set()
for i in range(0, len(input_string)):

    c = input_string[i]

    if c in sub_string:
        crnt_longest = ""
        sub_string.clear()

    crnt_longest = crnt_longest + c
    sub_string.add(c)

    if len(crnt_longest) > len(lngt_string):
        lngt_string = crnt_longest
print("Longest substring is", lngt_string)
print(" length is ", len(lngt_string))
