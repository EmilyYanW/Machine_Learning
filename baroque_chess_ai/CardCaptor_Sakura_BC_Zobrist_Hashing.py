'''CardCaptor_Sakura_BC_Zobrist_Hashing.py
This includes the implementation of Zobrist Hashing for the agent of Cardcaptor_Sakura.
'''

from random import randint

# Set up a 64x2 array of # random ints.
S = 64
P = 2
zobristnum = [[0]*P for i in range(S)]

zob_state_visited = {}


def init_hashtable():
    global zobristnum 
    for i in range(S):
        for j in range(P): 
            zobristnum[i][j]= randint(0, 4294967296)


# Hash the given state to an int. 
def zhash(state):
    global zobristnum 
    val = 0;
    board = state.board
    for row in range(8):
        for col in range(8):
            piece = None
            if(board[row][col] % 2 == 0): piece = 0 
            if(board[row][col] % 2 != 0): piece = 1 
            if(piece != None):
                idx = row * 8 + col
                val ^= zobristnum[idx][piece] 
    return val