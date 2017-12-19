
i = 1300.00
j = 300.9561

def getper(a,b):

    maxn= max(a,b)

    per = abs(a-b)/float(maxn)

    per = 100-(per*100)

    return per

p = getper(i,j)
print p