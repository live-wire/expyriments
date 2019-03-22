# Kanade's algorithm

# Maximum subarray sum


def mss(arr):
    
    sm = arr[0]  
    glob = arr[0]
    sireset = False
    si = 0

    for i in range(1, len(arr)):
        it = arr[i]
        sm = max(it, sm + it)
        glob = max(glob, sm)
    return glob

print(mss([-11, -2, -3, -20, -5, -6]))