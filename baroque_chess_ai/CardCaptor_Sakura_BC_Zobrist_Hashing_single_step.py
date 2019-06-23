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


def zhash_one_step(move, old_state, old_zhash):
    global zobristnum
    board = old_state.board

    new_zhash = old_zhash

    old_row = move[0][0]
    old_col = move[0][1]
    new_row = move[1][0]
    new_col = move[1][1]

    # deal with new position (moved-to position) captured side

    piece = None
    # if there is no piece at the new position then Piece = None
    if (board[new_row][new_col] % 2 == 0): piece = 0
    if (board[new_row][new_col] % 2 != 0): piece = 1
    if (piece != None):
        idx = new_row * 8 + new_col
        new_zhash ^= zobristnum[idx][piece]

    # deal with old and new position moving side piece
    piece = 0
    if (board[old_row][old_col] % 2 == 0):
        piece = 0
    else:
        piece = 1

    new_zhash ^= zobristnum[old_row * 8 + old_col][piece]
    new_zhash ^= zobristnum[new_row * 8 + new_col][piece]

    return new_zhash
