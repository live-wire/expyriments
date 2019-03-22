# Minimum moves to reach a spot on a chessboard by a Knight
from functools import reduce

def main():
    chessboard_size = 8
    chess_knights(chessboard_size)

def moves(a,b):
    return [(a+2, b+1), (a+2, b-1), (a-2, b+1), (a-2, b-1), (a+1, b+2), (a+1, b-2), (a-1, b+2), (a-1, b-2)]

def is_valid(a,b,n):
    return a>=0 and a < n and b>=0 and b< n

def spread(board, is_visited, a, b, distance_from_start = 0):
    if a>=0 and a < len(board) and b>=0 and b< len(board):
        if is_visited[a][b] is not True or distance_from_start < board[a][b]:
            board[a][b] = distance_from_start
            is_visited[a][b] = True
            for move in moves(a,b):
                spread(board, is_visited, move[0], move[1], distance_from_start+1)


def spread_optimized(board, is_visited, a, b, distance_from_start = 0):
    q = [(a,b, distance_from_start)]
    while len(q)!=0:
        a,b,d = q.pop()
        if is_valid(a,b,len(board)) and is_visited[a][b] is not True:
            board[a][b] = d
            is_visited[a][b] = True
            for move in moves(a,b):
                q.insert(0, (move[0], move[1], d+1))
    

def update(board, is_visited, a, b, distance_from_start):
    if a>=0 and a < len(board) and b>=0 and b< len(board):
        if is_visited[a][b] is not True:
            board[a][b] = distance_from_start
            is_visited[a][b] = True

def is_solved(is_visited):
    return reduce(lambda x,y: x and y, list(map(lambda a: reduce(lambda x,y: x and y, a), is_visited)))


def chess_knights(n, start_at=(0, 0)):
    board = [[-1]*n for i in range(n)]
    is_visited = [[False]*n for i in range(n)]
    x,y = start_at
    #spread(board, is_visited, x, y, 0)
    spread_optimized(board, is_visited, x, y, 0)
    print_board(board)

def print_board(board):
    n = len(board)
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=" ")
        print()
    print("\n")

if __name__=="__main__":
    main()