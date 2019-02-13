n=int(input("Enter number of Transactions"))
#n is the number of transactions
total=0
while n!=0:
    str =input("Type of Transaction:")
    t = str.split(" ")
    # type is to determine whether it's a deposit or withdrawal
    type = t[0]
    # amt is the amount of money deposited or withdrawn
    amt = int(t[1])
    if type == "Deposit" or type == "deposit":
        total += amt
    elif type =="Withdrawl" or type == "withdrawl":
        total -= amt
    else:
        pass
    n-=1
print("total amount = $",total)
