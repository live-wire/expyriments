# new year's chaos
def minimumBribes(q):
    bribes = 0
    for i,item in enumerate(reversed(q)):
        it = i+1
        if q[i] - (i+1) > 2:
            return 'Too chaotic'
        j = max(0, q[i]-2)
        print(i,j,q[i],q[j])
        while(j<i):
            if q[j]>q[i]:
                bribes = bribes + 1
            j = j+1
    return bribes

z = [1,2,5,3,7,8,6,4]
print(minimumBribes(z))