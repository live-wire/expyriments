# Sorts
def quicksort(arr):
    if len(arr)==0:
        return []

    pivot = arr[0]
    pivotPos = 0

    left = 0
    right = len(arr)-1

    while left < right:
        while left < pivotPos:
            if arr[left]>pivot:
                break
            left = left+1
        while right > pivotPos:
            if arr[right]<pivot:
                break
            right = right-1
        if left == pivotPos and right > left:
            pivotPos = right
        elif right == pivotPos and left < right:
            pivotPos = left
        if left < right:
            swap(arr, left, right)
    return quicksort(arr[:pivotPos]) + [pivot] + quicksort(arr[pivotPos+1:])

def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = int(len(arr)/2)
    left = arr[:mid]
    right = arr[mid:]

    mergesort(left)
    mergesort(right)
    i=0
    j=0
    k=0
    while i<len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i = i+1
        elif right[j] <= left[i]:
            arr[k] = right[j]
            j = j+1
        k = k+1

    while i<len(left):
        arr[k] = left[i]
        i = i+1
        k=k+1
    while j<len(right):
        arr[k] = right[j]
        j = j+1
        k = k+1
    return arr 


def main():
    arr = [11,22,3,5,1,6]
    # print(quicksort(arr))
    print(mergesort(arr))

if __name__ == "__main__":
    main()

