# Binary search replace



def binSearchPlace(arr, item):
    
    if len(arr) == 1:
        arr[0] = item
        return arr

    l = 0
    r = len(arr)
    m = (l+r)/2
    while(l<r):
        m = int((l+r)/2)
        if item >= arr[m]:
            l = m+1
        else:
            r = m
    

    if arr[m] < item:
        m = m+1
    print(arr, item, "<%d>"%m)
    arr[m] = item
    return arr

binSearchPlace([2], 3)
binSearchPlace([2,5], 3)
binSearchPlace([2,5,7], 6)
binSearchPlace([2,5,7,9], 1)
binSearchPlace([2,5,7,9], 6)

