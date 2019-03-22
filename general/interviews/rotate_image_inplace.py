# Rotate an image 2D array 90 degrees (inplace)

import numpy as np

def rotate90(img_flat):
    n = int(np.sqrt(len(img_flat)))

    img = []
    k=0
    for i in range(n):
        row = []
        for j in range(n):
            row.append(img_flat[k])
            k=k+1
        img.append(row)
    printImg(img, n)
    img = transpose(img, n)
    printImg(img, n)
    img = columnReverse(img, n)
    printImg(img, n)
    

def columnReverse(img, n):
    for j in range(int(n/2)):
        other = n-j-1
        for i in range(n):
            temp = img[i][j]
            img[i][j] = img[i][other]
            img[i][other] = temp
    return img



def transpose(img, n):
    # Find transpose first
    for i in range(n):
        for j in range(0, i):
            temp = img[i][j]
            img[i][j] = img[j][i]
            img[j][i] = temp
    return img

def flatten(arr, n):
    k = 0
    z = []
    for i in range(n):
        for j in range(n):
            z.append(arr[i][j])
    return z

def printImg(arr, n):
    for i in range(n):
        for j in range(n):
            print(arr[i][j],end=" ")
        print()
    print("\n")
rotate90([1,2,3,4,5,6,7,8,9])
