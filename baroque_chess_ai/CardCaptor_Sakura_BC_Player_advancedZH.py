'''CardCaptor_Sakura_BC_Player_advancedZH.py
The beginnings of an agent that might someday play Baroque Chess.
'''

import BC_state_etc as BC
import CardCaptor_Sakura_BC_Zobrist_Hashing_single_step as ZH
import copy
import time

PIECE_VALUES = {'k':100000, 'f': 800, 'i': 500, 'l': 400, 'w': 350, 'c': 300, 'p': 100, '-':0}
CURRENT_MOVE_START_TIME = 0;
TIME_LIMIT = 0;
END_TIME = None
CURRENT_STATE_STATIC_VAL = 0
N_STATES_EXPANDED = 0
N_STATIC_EVALS = 0
N_CUTOFFS = 0
N_ApplyZ = 0

DIALOGS = ["The chess created by Clow, abandon your old form and reincarnate! Under the name of your owner Sakura!",
            "I Sakura, command you under your contract! Release!",
           "The key which hides powers of the stars, show your true form before me!",
            "Watch out! Here is my move.", "Your move!", "Here you go!", "Here's my move."]

SPEC_DIALOGS = {'f': "Return to the guise you were meant to be in! Freezer!",
                'i': "Return to the guise you were meant to be in! Imitator!",
                'l': "Return to the guise you were meant to be in! Long Leaper!",
                'c': "Return to the guise you were meant to be in! Coordinator!"}


# Returns a dict object with the following four attributes
def parameterized_minimax(currentState, alphaBeta=False, ply=3,
    useBasicStaticEval=True, useZobristHashing=False):

    '''Implement this testing function for your agent's basic
    capabilities here.'''

    global CURRENT_STATE_STATIC_VAL
    global N_STATES_EXPANDED
    global N_STATIC_EVALS
    global N_CUTOFFS
    global N_ApplyZ

    CURRENT_STATE_STATIC_VAL = basicStaticEval(currentState) if useBasicStaticEval else staticEval(currentState)
    N_STATES_EXPANDED = 0
    N_STATIC_EVALS = 0
    N_CUTOFFS = 0

    old_hash = ZH.zhash(currentState)

    if not alphaBeta:
        minimax(currentState, old_hash, ply, currentState.whose_move, useBasicStaticEval)
    elif alphaBeta:
        minimax_with_pruning(currentState, old_hash, ply, -10000000, 10000000, currentState.whose_move, useBasicStaticEval, False)
    print("Accumulated Reused Number of Zobrist Hashing Number is: " + str(N_ApplyZ))
    result = {'CURRENT_STATE_STATIC_VAL': CURRENT_STATE_STATIC_VAL,
              'N_STATES_EXPANDED': N_STATES_EXPANDED,
              'N_STATIC_EVALS': N_STATIC_EVALS,
              'N_CUTOFFS': N_CUTOFFS,
              'ply': ply}
    return result


def makeMove(currentState, currentRemark, timelimit=10):

    """

    This function inputs a current state, and finds the optimal move based on Minimax

    RETURN:

    [[best_move_found, newState], newRemark]

    best_move_found: ((start i, start j), (end i, end j ))
    newState: a state instance
    newRemark: a string


    """
    global CURRENT_MOVE_START_TIME
    global TIME_LIMIT
    global END_TIME
    global N_ApplyZ
    global N_STATES_EXPANDED
    global N_STATIC_EVALS
    global N_CUTOFFS

    CURRENT_MOVE_START_TIME= time.time() # global variable to store the starting time at the begining of each move
    TIME_LIMIT = timelimit
    END_TIME = time.time() + TIME_LIMIT

    newState = BC.BC_state(currentState.board)
    depth = 0
    best_move_found = ()

    legalMoves = generateMoves(currentState)
    # print(legalMoves)

    #For test:
    #best_move_found = getRandomMoveForTest(legalMoves)
    #newState = changeState(currentState, best_move_found)

    current_hash = ZH.zhash(currentState)


    # Compute the new state for a move. An anytime algorithm based on IDDFS.
    if currentState.whose_move == 1: # white -> max
        best_move_value = -999999

        # while used time is less than 0.1 of total time
        # increment maximum depth


        while END_TIME - time.time() > TIME_LIMIT * 0.3:
            for move in legalMoves:
                tempState = changeState(currentState, move)
                tempState_hash = ZH.zhash_one_step(move, currentState, current_hash)

                value = minimax_with_pruning(tempState, tempState_hash, depth, -10000000, 10000000, tempState.whose_move, False, True)
                if value >= best_move_value:
                    best_move_value = value
                    best_move_found = move
                    newState = copy.deepcopy(tempState)

        depth += 1 # This makes IDDFS because each time we increment the maximum depth

    else: # black -> min
        best_move_value = 999999
        while END_TIME - time.time() > TIME_LIMIT * 0.3:
            for move in legalMoves:
                tempState = changeState(currentState, move)
                tempState_hash = ZH.zhash_one_step(move, currentState, current_hash)

                value = minimax_with_pruning(tempState, tempState_hash, depth, -10000000, 10000000, tempState.whose_move, False, True)
                if value <= best_move_value:
                    best_move_value = value
                    best_move_found = move # move format ((start i, start j), (end i, end j))
                    newState = copy.deepcopy(tempState)
        depth += 1

    #print("The best eval value is " + str(value))

    # Pring out Zobrist Hashing Reuse number
    print("Accumulated Reused Number of Zobrist Hashing Number is: " + str(N_ApplyZ))
    #print("Accumulated N_STATES_EXPANDED: " + str(N_STATES_EXPANDED))
    print("Accumulated N_STATIC_EVALS: " + str(N_STATIC_EVALS))
    #print("Accumulated N_CUTOFFS: " + str(N_CUTOFFS))
    # Make up a new remark
    piece_moved = BC.CODE_TO_INIT[newState.board[best_move_found[1][0]][best_move_found[1][1]]].lower()
    if piece_moved in SPEC_DIALOGS:
        newRemark = SPEC_DIALOGS[piece_moved]
    else:
        newRemark = choice(DIALOGS)

    return [[best_move_found, newState], newRemark]


def nickname():
    return "CardCaptor Sakura"


def introduce():
    return """I'm CardCaptor Sakura, a Baroque Chess agent! I am a ten-year-old student from the Japanese city of Tomoeda.
    I accidentally released a set of magical cards called Clow and I will now capture them.
    I was created by Liuqing Ma and Wendan Yan."""


def prepare(player2Nickname):
    ''' Here the game master will give your agent the nickname of
    the opponent agent, in case your agent can use it in some of
    the dialog responses.  Other than that, this function can be
    used for initializing data structures, if needed.'''
    # initialize the hash table
    ZH.init_hashtable()


def basicStaticEval(state):
    '''Use the simple method for state evaluation described in the spec.
    This is typically used in parameterized_minimax calls to verify
    that minimax and alpha-beta pruning work correctly.'''
    white_value = 0
    black_value = 0
    for r in range(8):
        for c in range(8):
            piece = state.board[r][c]
            if piece == 0:
                continue
            elif piece % 2 == 1: # white
                if piece == 3:
                    white_value += 1
                elif piece == 13:
                    white_value += 100
                else:
                    white_value += 2
            else:  # black
                if piece == 2:
                    black_value -= 1
                elif piece == 12:
                    black_value -= 100
                else:
                    black_value -= 2
    return white_value + black_value

def staticEval(state):
    '''Compute a more thorough static evaluation of the given state.
    This is intended for normal competitive play.  How you design this
    function could have a significant impact on your player's ability
    to win games.'''
    white_value = 0
    black_value = 0
    for r in range(8):
        for c in range(8):
            piece = state.board[r][c]
            if piece % 2 == 1: # white
                white_value += PIECE_VALUES[BC.CODE_TO_INIT[piece].lower()]

                if piece == 3: # pincer white
                    white_value += pincerEvalWhite[r][c] + 10
                if piece == 5 or piece == 15: # coordinator or freezer white
                    white_value += coordinatorEvalWhite[r][c] + 50
                if piece == 7: # long Leaper white
                    white_value += longLeaperEval[r][c] + 30
                if piece == 9: # imitator white
                    white_value += imitatorEvalWhite[r][c] + 30
                if piece == 11: # withdrawer white
                    white_value += evalWithdrawer[r][c] + 90
                if piece == 13: # king white
                    white_value += kingEvalWhite[r][c] + 900

            else:  # black
                black_value -= PIECE_VALUES[BC.CODE_TO_INIT[piece].lower()]

                if piece == 2: # pincer black
                    black_value += pincerEvalBlack[r][c] + 10
                if piece == 4 or piece == 14: # coordinator or freezer black
                    black_value += coordinatorEvalBlack[r][c] + 50
                if piece == 6: # long Leaper black
                    black_value += longLeaperEval[r][c] + 30
                if piece == 8: # imitator black
                    black_value += imitatorEvalBlack[r][c] + 30
                if piece == 10: # withdrawer black
                    black_value += evalWithdrawer[r][c] + 90
                if piece == 12: # king black
                    black_value += kingEvalBlack[r][c] + 900
    return white_value + black_value


def minimax(currentState, old_hash, depth, isMaximisingPlayer, useBasicStaticEval):

    """
    A minimax search with static evaluation of every successor
    Assuming that the opponent is playing rationally

    """
    global N_STATIC_EVALS
    global N_STATES_EXPANDED
    global N_ApplyZ


    nbs, moves = neighbors(currentState)


    if depth == 0 or len(nbs) == 0:
        N_STATIC_EVALS += 1
        if (old_hash, depth) in ZH.zob_state_visited:
            N_ApplyZ += 1
            return ZH.zob_state_visited[(old_hash, depth)]
        else:
            if useBasicStaticEval:
                value = basicStaticEval(currentState)
                ZH.zob_state_visited[(old_hash, depth)] = value
                return value
            else:
                value = staticEval(currentState)
                ZH.zob_state_visited[(old_hash, depth)] = value
                return value


    # return the states after one legal move

    if isMaximisingPlayer == 1:
        v = -10000000
        for i,child in enumerate(nbs):
            N_STATES_EXPANDED += 1
            child_move = moves[i]
            child_hash = ZH.zhash_one_step(child_move, currentState, old_hash)

            if (child_hash, depth-1) in ZH.zob_state_visited:
                b = ZH.zob_state_visited[(child_hash, depth-1)]
                N_ApplyZ += 1
            else:
                b = minimax(child, child_hash, depth - 1, 1-isMaximisingPlayer, useBasicStaticEval)
            v = max(v,b)
        return v
    else:
        v = 10000000
        for i, child in enumerate(nbs):
            N_STATES_EXPANDED += 1
            child_move = moves[i]
            child_hash = ZH.zhash_one_step(child_move, currentState, old_hash)

            if (child_hash, depth-1) in ZH.zob_state_visited:
                b = ZH.zob_state_visited[(child_hash, depth-1)]
                N_ApplyZ += 1
            else:
                b = minimax(child, child_hash, depth - 1, 1-isMaximisingPlayer, useBasicStaticEval)
            v = min(v,b)
        return v


# A minimax search with alpha-beta pruning
def minimax_with_pruning(currentState, old_hash, depth, alpha, beta, isMaximisingPlayer, useBasicStaticEval, limitTime=False):
    global N_STATIC_EVALS
    global N_STATES_EXPANDED
    global N_CUTOFFS
    global END_TIME
    global N_ApplyZ

    nbs, moves = neighbors(currentState)

    if depth == 0 or len(nbs) == 0:
        N_STATIC_EVALS += 1
        # if ZH.zhash(currentState) in ZH.zob_state_visited:
        if (old_hash, depth) in ZH.zob_state_visited:
            N_ApplyZ += 1
            # print("revist")

            return ZH.zob_state_visited[(old_hash, depth)]
        else:
            if useBasicStaticEval:
                value = basicStaticEval(currentState)
                ZH.zob_state_visited[(old_hash, depth)] = value
                return value
            else:
                value = staticEval(currentState)
                ZH.zob_state_visited[(old_hash, depth)] = value
                return value

    N_STATES_EXPANDED += 1

    if isMaximisingPlayer == 1:
        v = -10000000
        for i, child in enumerate(nbs):
            if limitTime and END_TIME - time.time() < TIME_LIMIT * 0.2:
                return
            child_move = moves[i]
            child_hash = ZH.zhash_one_step(child_move, currentState, old_hash)

            if (child_hash, depth-1) in ZH.zob_state_visited:
                b = ZH.zob_state_visited[(child_hash, depth-1)]
                N_ApplyZ += 1
                #print("revist")
            else:
                b = minimax_with_pruning(child, child_hash, depth - 1, alpha, beta, 1-isMaximisingPlayer, useBasicStaticEval, limitTime)
                ZH.zob_state_visited[(child_hash, depth-1)] = b
            v = max(v,b)
            alpha = max(alpha, b)
            if beta <= alpha:
                N_CUTOFFS += 1
                break
        return v
    else:
        v = 10000000
        for i,child in enumerate(nbs):
            if limitTime and END_TIME - time.time() < TIME_LIMIT * 0.2:
                return
            child_move = moves[i]
            child_hash = ZH.zhash_one_step(child_move, currentState, old_hash)

            if (child_hash, depth-1) in ZH.zob_state_visited:
                b = ZH.zob_state_visited[(child_hash, depth-1)]
                N_ApplyZ += 1
                #print("revist")
            else:
                b = minimax_with_pruning(child, child_hash, depth - 1, alpha, beta, 1-isMaximisingPlayer, useBasicStaticEval,limitTime)
                ZH.zob_state_visited[(child_hash, depth-1)] = b
            v = min(v,b)
            beta = min(beta, b)
            if beta <= alpha:
                N_CUTOFFS += 1
                break
        return v


# Generates a list of all legal moves given the current state.
def generateMoves(currentState):
    moves = []
    for r in range(0, 8):
        for c in range(0, 8):
            piece = BC.CODE_TO_INIT[currentState.board[r][c]]
            if piece != '-':
                is_white_piece = piece.isupper()
                if (currentState.whose_move and is_white_piece) or \
                        (not currentState.whose_move and not is_white_piece):
                    frozen = freezeCheck(currentState.board, r, c, currentState.whose_move)
                    if not frozen:
                        getMoves(currentState.board, piece, r, c, moves)
    moves = list(set(moves))

    moves = sorted(moves, key=lambda move: move[0][0], reverse=True)
    moves = sorted(moves, key=lambda move: move[0][1])
    moves = sorted(moves, key=lambda move: move[1][0], reverse=True)
    moves = sorted(moves, key=lambda move: move[1][1])

    return moves


# Returns whether the piece at the given coordinates is frozen.
def freezeCheck(board, r, c, white_to_move):
    opp_freezer = 14 if white_to_move == 1 else 15  # if it is surrounded by opposite 'F' or 'f'
    is_freezer = (white_to_move == 1 and board[r][c] == 15) or (not white_to_move and board[r][c] == 14)
    opp_imitator = 8 if white_to_move == 1 else 9  # if itself is freezer and  surrounded by opposite 'I' or 'i'
    for i in range(3):
        for j in range(3):
            rank = r - 1 + i
            file = c - 1 + j
            if 0 <= rank <= 7 and 0 <= file <= 7:
                if board[rank][file] == opp_freezer or (is_freezer and board[rank][file] == opp_imitator):
                    return True
    return False


# Returns a list of all legal moves for that piece, including captures.
def getMoves(board, piece, r, c, moveList=[]):
    if piece.lower() == 'p':
        getRookLikeMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'f':
        getQueenLikeMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'l':
        getQueenLikeMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'i':
        getQueenLikeMoves(piece, board, r, c, moveList)
        # special case: Can move and capture adjacent enemy King:
        imitateKingCaptureMove(board, r, c, moveList)
    elif piece.lower() == 'w':
        getQueenLikeMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'k':
        getKingMoves(board, r, c, moveList)
    elif piece.lower() == 'c':
        getQueenLikeMoves(piece, board, r, c, moveList)


def imitateKingCaptureMove(board, r, c, moveList):
    for i in range(3):
        for j in range(3):
            rank = r - 1 + i
            file = c - 1 + j
            if 0 <= rank <= 7 and 0 <= file <= 7:
                adj_king = board[rank][file] == 12 or board[rank][file] == 13
                if isOppositePiece(board, r, c, rank, file) and adj_king:
                    moveList.append(((r, c), (rank, file)))


# Returns a list of available Rook-like moves for the current piece.
# Moves horizontally or vertically
def getRookLikeMoves(piece, board, r, c, moveList):
    south = True
    north = True
    east = True
    west = True

    for i in range(1, 8):
        # if we hit a piece, stop or if we hit the edge
        if r + i <= 7 and south:  # going south
            if board[r + i][c] == 0:
                moveList.append(((r, c), (r + i, c)))
            elif isOppositePiece(board, r, c, r+i, c) and r + i <= 6 and board[r+i+1][c] == 0:
                if piece.lower() == 'l'\
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r+i][c]].lower() == 'l':
                    moveList.append(((r, c), (r + i + 1, c)))
                    south = False
                else:
                    south = False

            else:
                south = False
        if r - i >= 0 and north:  # going north
            if board[r - i][c] == 0:
                moveList.append(((r, c), (r - i, c)))
            elif isOppositePiece(board, r, c, r-i, c) and r - i >= 1 and board[r - i - 1][c] == 0:
                if piece.lower() == 'l'\
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r-i][c]].lower() == 'l':
                    moveList.append(((r, c), (r - i - 1,c)))
                    north = False
                else:
                    north = False
            else:
                north = False
        if c + i <= 7 and east:  # going east
            if board[r][c + i] == 0:
                moveList.append(((r, c), (r, c + i)))
            elif isOppositePiece(board, r, c, r, c+i) and c + i <= 6 and board[r][c + i + 1] == 0:
                if piece.lower() == 'l' \
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r][c+i]].lower() == 'l':
                    moveList.append(((r, c), (r,c+i+1)))
                    east = False
                else:
                    east = False
            else:
                east = False
        if c - i >= 0 and west:  # going west
            if board[r][c - i] == 0:
                moveList.append(((r, c), (r, c - i)))
            elif isOppositePiece(board, r, c, r, c-i) and c - i >= 1 and board[r][c-i-1] == 0:
                if piece.lower() == 'l' \
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r][c-i]].lower() == 'l':
                    moveList.append(((r, c), (r, c-i-1)))
                    west = False
                else:
                    west = False
            else:
                west = False


# Returns a list of available Bishop-like moves (diagonal movements) for the current piece
# Moves diagonally
def getBishopLikeMoves(piece, board, r, c, moveList):
    ne = True
    nw = True
    se = True
    sw = True

    for i in range(1, 8):
        if r + i <= 7 and c + i <= 7 and se:  # going southeast
            if board[r + i][c + i] == 0:
                moveList.append(((r, c), (r + i, c + i)))
            elif isOppositePiece(board, r, c, r+i, c+i) and r + i <= 6 and c + i <= 6 and board[r + i + 1][c + i + 1] == 0:
                if piece.lower() == 'l' \
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r + i][c + i]].lower() == 'l':
                    moveList.append(((r, c), (r + i + 1, c + i + 1)))
                    se = False
                else:
                    se = False
            else:
                se = False
        if r - i >= 0 and c - i >= 0 and nw:  # going northwest
            if board[r - i][c - i] == 0:
                moveList.append(((r, c), (r - i, c - i)))
            elif isOppositePiece(board, r, c, r-i, c-i) and r - i >= 1 and c - i >= 1 and board[r - i -1][c- i - 1] == 0:
                if piece.lower() == 'l'\
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r - i][c - i]].lower() == 'l':
                    moveList.append(((r, c), (r - i - 1, c - i - 1)))
                    nw = False
                else:
                    nw = False
            else:
                nw = False
        if c + i <= 7 and r - i >= 0 and ne:  # going northeast
            if board[r - i][c + i] == 0:
                moveList.append(((r, c), (r - i, c + i)))
            elif isOppositePiece(board, r, c, r-i, c+i) and r - i >= 1 and c + i <= 6 and board[r - i - 1][c + i + 1] == 0:
                if piece.lower() == 'l'\
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r - i][c + i]].lower() == 'l':
                    moveList.append(((r,c), (r - i - 1, c + i + 1)))
                    ne = False
                else:
                    ne = False
            else:
                ne = False
        if c - i >= 0 and r + i <= 7 and sw:  # going southwest
            if board[r + i][c - i] == 0:
                moveList.append(((r, c), (r + i, c - i)))
            elif isOppositePiece(board, r, c, r+i, c-i) and r + i <= 6 and c - i >= 1 and board[r+i+1][c - i - 1] == 0:
                if piece.lower() == 'l' \
                    or piece.lower() == 'i' and BC.CODE_TO_INIT[board[r + i][c-i]].lower() == 'l':
                    moveList.append(((r, c), (r + i + 1, c - i - 1)))
                    sw = False
                else:
                    sw = False
            else:
                sw = False


# Returns a list of available Queen-like moves for the current piece.
# Moves vertically, horizontally or diagonally
def getQueenLikeMoves(piece, board, r, c, moveList):
    getRookLikeMoves(piece, board, r, c, moveList)
    getBishopLikeMoves(piece, board, r, c, moveList)


# Returns a list of available King moves for the current piece. A king can move one square in any direction (horizontally, vertically, or diagonally)
# Including captures
def getKingMoves(board, r, c, moveList):
    for i in range(3):
        for j in range(3):
            rank = r - 1 + i
            file = c - 1 + j
            if 0 <= rank <= 7 and 0 <= file <= 7 \
                    and board[r][c] % 2 != board[rank][file] % 2:  # King cannot move into friendlies
                moveList.append(((r, c), (rank, file)))

# Returns whether two pieces are opposite
def isOppositePiece(board, r1, c1, r2, c2):
    return board[r1][c1] != 0 and board[r2][c2] != 0 and board[r1][c1] % 2 != board[r2][c2] % 2


# Returns a list of new states after legal moves applied
def neighbors(state):
    result = []
    copy_state = copy.deepcopy(state)
    moves = generateMoves(state)
    for move in moves:
        result.append(changeState(copy_state, move))
    return result, moves


# Returns an updated state after making the given move.
def changeState(state, move):
    new_state = copy.deepcopy(state)
    old_r = move[0][0]
    old_c = move[0][1]
    new_r = move[1][0]
    new_c = move[1][1]
    piece = state.board[old_r][old_c]
    new_state.board[old_r][old_c] = 0
    new_state.board[new_r][new_c] = piece
    # Handle special capture cases
    handleSpecialCaptures(state, new_state, piece, old_r, old_c, new_r, new_c)

    new_state.whose_move ^= 1
    return new_state


def handleSpecialCaptures(state, new_state, piece, old_r, old_c, new_r, new_c):

    # when the new position on the old board is vacent, it might be special capture
    if state.board[new_r][new_c] == 0:

        " if the piece is Withdrawer moving "
        if piece == 10 or piece == 11:
            withdrawerCapture(state, new_state, old_r, old_c, new_r, new_c)

        "if the piece is Long Leaper moving "
        if piece == 6 or piece == 7:
            leaperCapture(state, new_state, old_r, old_c, new_r, new_c)

        "if the piece is Pincer moving"
        if piece == 2 or piece == 3:
            pincerCapture(state, new_state, old_r, old_c, new_r, new_c)

        "if the piece is Coordinator moving"
        if piece == 4 or piece == 5:
            coordinatorCapture(state, new_state, old_r, old_c, new_r, new_c)

        "if the piece is Imitator moving"
        if piece == 8 or piece == 9:
            imitatorCapture(new_state, old_r, old_c, new_r, new_c)


def withdrawerCapture(state, new_state, old_r, old_c, new_r, new_c):
    # moving southeast:
    if new_r > old_r and new_c > old_c and old_r-1 >=0 and old_c -1 >=0 and \
            isOppositePiece(state.board, old_r, old_c, old_r-1, old_c-1):
        new_state.board[old_r-1][old_c-1] = 0

    # moving northeast:
    if new_r < old_r and new_c > old_c and old_r + 1 <=7 and old_c-1 >=0 and \
            isOppositePiece(state.board, old_r, old_c, old_r+1, old_c-1):
        new_state.board[old_r+1][old_c-1] = 0

    # moving southwest:
    if new_r > old_r and new_c < old_c and old_r -1 >=0 and old_c +1 <= 7 and \
            isOppositePiece(state.board, old_r, old_c, old_r-1, old_c+1):
        new_state.board[old_r-1][old_c+1] = 0

    # moving norththwest:
    if new_r < old_r and new_c < old_c and old_r+1 <= 7 and old_c + 1 <= 7 and \
            isOppositePiece(state.board, old_r, old_c, old_r+1, old_c+1):
        new_state.board[old_r+1][old_c+1] = 0

    # moving south:
    if new_r > old_r and new_c ==  old_c and old_r-1 >= 0 and \
            isOppositePiece(state.board, old_r, old_c, old_r-1, old_c):
        new_state.board[old_r-1][old_c] = 0

    # moving north:
    if new_r < old_r and new_c == old_c and old_r+1 <= 7 and \
            isOppositePiece(state.board, old_r, old_c, old_r+1, old_c):
        new_state.board[old_r+1][old_c] = 0

    # moving east:
    if new_r == old_r and new_c > old_c and old_c-1 >= 0 and \
            isOppositePiece(state.board, old_r, old_c, old_r, old_c-1):
        new_state.board[old_r][old_c-1] = 0

    # moving west:
    if new_r == old_r and new_c < old_c and old_c+1 <= 7 and \
            isOppositePiece(state.board, old_r, old_c, old_r, old_c+1):
        new_state.board[old_r][old_c+1] = 0


def leaperCapture(state, new_state, old_r, old_c, new_r, new_c):
    # moving southeast:
    if new_r > old_r and new_c > old_c and isOppositePiece(new_state.board, new_r,new_c, new_r-1, new_c-1):
        new_state.board[new_r-1][new_c-1] = 0

    # moving northeast:
    if new_r < old_r and new_c > old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c-1):
        new_state.board[new_r+1][new_c-1] = 0

    # moving southwest:
    if new_r > old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c+1):
        new_state.board[new_r-1][new_c+1] = 0

    # moving norththwest:
    if new_r < old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c+1):
        new_state.board[new_r+1][new_c+1] = 0

    # moving south:
    if new_r > old_r and new_c ==  old_c and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c):
        new_state.board[new_r-1][new_c] = 0

    # moving north:
    if new_r < old_r and new_c == old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c):
        new_state.board[new_r+1][new_c] = 0

    # moving east:
    if new_r == old_r and new_c > old_c and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-1):
        new_state.board[new_r][new_c-1] = 0

    # moving west:
    if new_r == old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+1):
        new_state.board[new_r][new_c+1] = 0


def pincerCapture(state, new_state, old_r, old_c, new_r, new_c):
    # north
    if new_r-2 >= 0 and state.board[new_r-2][new_c] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r-2, new_c) and \
                isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c):

                new_state.board[new_r-1][new_c] = 0

    # south
    if new_r+2 <= 7 and state.board[new_r+2][new_c] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r+2, new_c) and \
                isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c):

                new_state.board[new_r+1][new_c] = 0

    # east
    if new_c+2 <= 7 and state.board[new_r][new_c+2] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+2) and \
                isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+1):

                new_state.board[new_r][new_c+1] = 0

    # west
    if new_c-2 >= 0 and state.board[new_r][new_c-2] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-2) and \
                isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-1):

                new_state.board[new_r][new_c-1] = 0


def coordinatorCapture(state, new_state, old_r, old_c, new_r, new_c):
    rank = new_r
    file = new_c
    found_k = False

    # find the rank and file of the King of the same player
    for i in range(8):
        for j in range(8):
            if state.board[i][j] == 12 or state.board[i][j] == 13 and \
                not isOppositePiece(new_state.board, new_r, new_c, i, j):
                rank = i
                file = j
                found_k = True
                break

    # find the diagonal rank and file of the opponent
    if found_k:
        corner1 = new_state.board[rank][new_c]
        if corner1 != 0 and isOppositePiece(new_state.board, new_r, new_c, rank, new_c):
            new_state.board[rank][new_c] = 0

        corner2 = new_state.board[new_r][file]
        if corner2 != 0 and isOppositePiece(new_state.board, new_r, new_c, new_r, file):
            new_state.board[new_r][file] = 0


def imitatorCapture(new_state, old_r, old_c, new_r, new_c):

    "acting like a Withdrawer"
    # moving southeast:
    if new_r > old_r and new_c > old_c and old_r-1 >=0 and old_c -1 >=0 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r-1, old_c-1) and \
                (new_state.board[old_r-1][old_c-1] == 10 or new_state.board[old_r-1][old_c-1] == 11):
        new_state.board[old_r-1][old_c-1] = 0

    # moving northeast:
    if new_r < old_r and new_c > old_c and old_r + 1 <=7 and old_c-1 >=0 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r+1, old_c-1) and \
                (new_state.board[old_r+1][old_c-1] == 10 or new_state.board[old_r+1][old_c-1] == 11):
        new_state.board[old_r+1][old_c-1] = 0

    # moving southwest:
    if new_r > old_r and new_c < old_c and old_r -1 >=0 and old_c +1 <= 7 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r-1, old_c+1) and \
                (new_state.board[old_r-1][old_c+1] == 10 or new_state.board[old_r-1][old_c+1] == 11):
        new_state.board[old_r-1][old_c+1] = 0

    # moving norththwest:
    if new_r < old_r and new_c < old_c and old_r+1 <= 7 and old_c + 1 <= 7 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r+1, old_c+1) and \
                (new_state.board[old_r+1][old_c+1] == 10 or new_state.board[old_r+1][old_c+1] == 11):
        new_state.board[old_r+1][old_c+1] = 0

    # moving south:
    if new_r > old_r and new_c ==  old_c and old_r-1 >= 0 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r-1, old_c) and \
                (new_state.board[old_r-1][old_c] == 10 or new_state.board[old_r-1][old_c] == 11):
        new_state.board[old_r-1][old_c] = 0

    # moving north:
    if new_r < old_r and new_c == old_c and old_r+1 <= 7 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r+1, old_c) and \
                (new_state.board[old_r+1][old_c] == 10 or new_state.board[old_r+1][old_c] == 11):
        new_state.board[old_r+1][old_c] = 0

    # moving east:
    if new_r == old_r and new_c > old_c and old_c-1 >= 0 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r, old_c-1) and \
                (new_state.board[old_r][old_c-1] == 10 or new_state.board[old_r][old_c-1] == 11):
        new_state.board[old_r][old_c-1] = 0

    # moving west:
    if new_r == old_r and new_c < old_c and old_c+1 <= 7 and \
            isOppositePiece(new_state.board, new_r, new_c, old_r, old_c+1) and \
                (new_state.board[old_r][old_c+1] == 10 or new_state.board[old_r][old_c+1] == 11):
        new_state.board[old_r][old_c+1] = 0


    "acting like a Long Leaper"

    # moving southeast:
    if new_r > old_r and new_c > old_c and isOppositePiece(new_state.board, new_r,new_c, new_r-1, new_c-1) and \
        ( new_state.board[new_r-1][new_c-1] == 6 or new_state.board[new_r-1][new_c-1] == 7):
        new_state.board[new_r-1][new_c-1] = 0

    # moving northeast:
    if new_r < old_r and new_c > old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c-1) and \
        ( new_state.board[new_r+1][new_c-1] == 6 or new_state.board[new_r+1][new_c-1] == 7):
        new_state.board[new_r+1][new_c-1] = 0

    # moving southwest:
    if new_r > old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c+1) and \
        ( new_state.board[new_r-1][new_c+1] == 6 or new_state.board[new_r-1][new_c+1] == 7):
        new_state.board[new_r-1][new_c+1] = 0

    # moving norththwest:
    if new_r < old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c+1) and \
        ( new_state.board[new_r+1][new_c+1] == 6 or new_state.board[new_r+1][new_c+1] == 7):
        new_state.board[new_r+1][new_c+1] = 0

    # moving south:
    if new_r > old_r and new_c ==  old_c and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c) and \
        ( new_state.board[new_r-1][new_c] == 6 or new_state.board[new_r-1][new_c] == 7):
        new_state.board[new_r-1][new_c] = 0

    # moving north:
    if new_r < old_r and new_c == old_c and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c) and \
        ( new_state.board[new_r+1][new_c] == 6 or new_state.board[new_r+1][new_c] == 7):
        new_state.board[new_r+1][new_c] = 0

    # moving east:
    if new_r == old_r and new_c > old_c and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-1) and \
        ( new_state.board[new_r][new_c-1] == 6 or new_state.board[new_r][new_c-1] == 7):
        new_state.board[new_r][new_c-1] = 0

    # moving west:
    if new_r == old_r and new_c < old_c and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+1) and \
        ( new_state.board[new_r][new_c+1] == 6 or new_state.board[new_r][new_c+1] == 7):
        new_state.board[new_r][new_c+1] = 0


    "acting like a Pincer moving"
    # if moving is vertical / horizontal
    if ( (new_r - old_r != 0) and (new_c - old_c == 0) ) or ( (new_c - old_c != 0) and (new_r - old_r == 0) ):

        # if there is a pincer at the right position of any of the four directions:

        if (new_r-2 >= 0 and (new_state.board[new_r-1][new_c] == 2 or new_state.board[new_r-1][new_c] == 3) and new_state.board[new_r-2][new_c] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r-2, new_c) and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c)) or \
            ( new_r+2 <= 7 and (new_state.board[new_r+1][new_c] == 2 or new_state.board[new_r+1][new_c] == 3) and new_state.board[new_r+2][new_c] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r+2, new_c) and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c)) or \
            ( new_c+2 <= 7 and (new_state.board[new_r][new_c+1] == 2 or new_state.board[new_r][new_c+1] == 3) and new_state.board[new_r][new_c+2] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+2) and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+1)) or \
            (new_c-2 >= 0 and (new_state.board[new_r][new_c-1] == 2 or new_state.board[new_r][new_c-1] == 3) and new_state.board[new_r][new_c-2] != 0 and \
            not isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-2) and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-1)):

            # north
            if new_r-1 >= 0 and isOppositePiece(new_state.board, new_r, new_c, new_r-1, new_c) and \
                new_r-2 >= 0 and new_state.board[new_r-2][new_c] != 0 and not isOppositePiece(new_state.board,
                new_r, new_c, new_r-2, new_c) :

                new_state.board[new_r-1][new_c] = 0

            # south
            if new_r+1 <= 7 and isOppositePiece(new_state.board, new_r, new_c, new_r+1, new_c) and \
                new_r+2 <= 7 and new_state.board[new_r+2][new_c] != 0 and not isOppositePiece(new_state.board,
                new_r, new_c, new_r+2, new_c):
                new_state.board[new_r+1][new_c] = 0

            # east
            if new_c+1 <= 7 and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c+1) and \
                new_c+2 <= 7 and new_state.board[new_r][new_c+2] != 0 and not isOppositePiece(new_state.board,
                new_r, new_c, new_r, new_c+2) :
                new_state.board[new_r][new_c+1] = 0

            # west
            if new_c-1 >= 0 and isOppositePiece(new_state.board, new_r, new_c, new_r, new_c-1) and \
                new_c-2 >= 0 and new_state.board[new_r][new_c-2] != 0 and not isOppositePiece(new_state.board,
                new_r, new_c, new_r, new_c-2) :
                new_state.board[new_r][new_c-1] = 0


    "acting like a Coordinator moving "
    rank = new_r
    file = new_c
    found_k = False

    # find the rank and file of the King of the same player
    for i in range(8):
        for j in range(8):

            if new_state.board[i][j] == 12 or new_state.board[i][j] == 13 and \
                not isOppositePiece(new_state.board, new_r, new_c, i, j):
                found_k = True
                rank = i
                file = j
                break

    # find the diagonal rank and file of the opponent
    if found_k == True:

        corner1 = new_state.board[rank][new_c]
        if corner1 != 0 and isOppositePiece(new_state.board, new_r, new_c, rank, new_c) and \
            ( new_state.board[rank][new_c] == 4 or new_state.board[rank][new_c] == 5):
            new_state.board[rank][new_c] = 0

        corner2 = new_state.board[new_r][file]
        if corner2 != 0 and isOppositePiece(new_state.board, new_r, new_c, new_r, file) and \
            ( new_state.board[new_r][file] == 4 or new_state.board[new_r][file] == 5):
            new_state.board[new_r][file] = 0


def timeIsUp():
    global END_TIME
    return END_TIME - time.time() <= 0.08


from random import choice
def getRandomMoveForTest(legalMoves):
    return choice(legalMoves)


pincerEvalWhite = \
    [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
        [1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
        [0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
        [0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
        [0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
        [0.5,  1.0, 1.0,  -2.0, -2.0,  1.0,  1.0,  0.5],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
    ]

pincerEvalBlack = [i[::-1] for i in pincerEvalWhite[::-1]]

longLeaperEval = \
    [
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
        [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
        [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
        [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
        [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
        [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
        [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
    ]

imitatorEvalWhite = [
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
    [ -1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
    [ -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
    [ -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
    [ -1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
    [ -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
]

imitatorEvalBlack = [i[::-1] for i in imitatorEvalWhite[::-1]]

coordinatorEvalWhite = [
    [  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [  0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [ -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
    [  0.0,   0.0, 0.0,  0.5,  0.5,  0.0,  0.0,  0.0]
]

coordinatorEvalBlack = [i[::-1] for i in coordinatorEvalWhite[::-1]]

evalWithdrawer =\
    [
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
    [ -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [  0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
    [ -1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
    [ -1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
    [ -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
]

kingEvalWhite = [

    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
    [ -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
    [ -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
    [  2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0 ],
    [  2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0 ]
]

kingEvalBlack = [i[::-1] for i in kingEvalWhite[::-1]]
