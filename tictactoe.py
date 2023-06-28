"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_entries = 0
    o_entries = 0
    if terminal(board):
        return "Terminal"
    else:
        for list in board:
            for entry in list:
                if entry == X:
                    x_entries += 1
                if entry == O:
                    o_entries += 1
        if o_entries == x_entries:
            return "X"
        else:
            return "O"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if terminal(board):
        return "Terminal"
    else:
        result = set()
        row = -1
        for list in board:
            row += 1
            col = -1
            for entry in list:
                col += 1
                if entry == EMPTY:
                    result.add((row, col))
        return result


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if not(action in actions(board)):
        raise Exception("Invalid Action")
    else:
        i = action[0]
        j = action[1]
        new_board = copy.deepcopy(board)
        current_player = player(board)
        if current_player == X:
            new_board[i][j] = "X"
        else:
            new_board[i][j] = "O"
        return new_board  


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for num in range(3):
        if (board[num][0] == board[num][1] == board[num][2]) and (board[num][0] != None): #row check
            return board[num][0]
        if (board[0][num] == board[1][num] == board[2][num]) and (board[0][num] != None): #col check
            return board[0][num]
    if (board[0][0] == board[1][1] == board[2][2]) and (board[0][0] != None): #diag check
        return board[0][0]
    if (board[0][2] == board[1][1] == board[2][0]) and (board[0][2] != None): #diag check
        return board[0][2]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    else:
        for list in board:
            for entry in list:
                if entry == EMPTY:
                    return False
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == "X":
        return 1
    elif winner(board) == "O":
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """ 
    def max_value(board):
        v = -math.inf
        if terminal(board):
            return utility(board)
        else:
            for action in actions(board):
                v = max(v, min_value(result(board, action)))
            return v

    def min_value(board):
        v = math.inf
        if terminal(board):
            return utility(board)
        else:
            for action in actions(board):
                v = min(v, max_value(result(board, action)))
            return v
       
    if terminal(board):
        return None
    else:
        current_player = player(board)
        if current_player == "X": #maximising
            optimal_action = None
            v = -math.inf
            for action in actions(board):
                current_utility = min_value(result(board, action))
                if current_utility > v:
                    v = current_utility
                    optimal_action = action
            return optimal_action
        else: #minimising
            optimal_action = None
            v = math.inf
            for action in actions(board):
                current_utility = max_value(result(board, action))
                if current_utility < v:
                    v = current_utility
                    optimal_action = action
            return optimal_action