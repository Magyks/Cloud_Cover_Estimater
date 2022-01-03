array = [[1 for i in range(20)] for j in range(20)]

def search(array,point):
    start=2
    for i in range(5):
        for j in range(-int(start/2),int(start/2)):
            x = point[0] + j
            y = point[1] - (5-i)
            if (0 <= x <= len(array[0])) and ( 0 <= y <= len(array)):
                array[y][x] = 0
        start += 2

    for i in range(5):
        start -= 2
        for j in range(-int(start/2),int(start/2)):
            x = point[0] + j
            y = point[1] + i
            if (0 <= x <= len(array[0])) and ( 0 <= y <= len(array)):
                array[y][x] = 0

    print(array)

def Average(array,point,size = 5):
    value = 0
    start=2
    for i in range(size):
        for j in range(-int(start/2),int(start/2)):
            x = point[0] + j
            y = point[1] - (size-i)
            if (0 <= x <= len(array[0])) and ( 0 <= y <= len(array)):
                value += array[y][x] 
        start += 2
        
    for i in range(size):
        start -= 2
        for j in range(-int(start/2),int(start/2)):
            x = point[0] + j
            y = point[1] + i
            if (0 <= x <= len(array[0])) and ( 0 <= y <= len(array)):
                value += array[y][x] 

    return value

search(array,(10,10))