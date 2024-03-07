import math
import copy
import random

def max_player(board,depth,  alpha , beta):
    if board.check_win(2):
        return -1
    elif board.turn==42 or depth==0:
        return 0
    
    v = -math.inf
    for action in board.not_full():
            new_board = copy.deepcopy(board)
            new_board.play(action)
            v = max(v, min_player(new_board, depth-1, alpha, beta ))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
    return v

def min_player(board, depth, alpha , beta):
    if board.check_win(1):
        return 1
    elif board.turn==42 or depth ==0:
        return 0

    v = math.inf
    for action in board.not_full():
        new_board = copy.deepcopy(board)
        new_board.play(action)
        v = min(v, max_player(new_board, depth-1, alpha ,beta ))
        beta = min(beta, v)
        if alpha >=beta:
            break
    return v


def alphaBeta(board, depth):
    if board.check_win(1) or board.check_win(2) or board.turn==42:
        return None 

    if board.whos_turn ==1:
        v = -math.inf
        best_action = [] 
        for action in board.not_full():
            new_board = copy.deepcopy(board) 
            new_board.play(action)
            next_value =  min_player(new_board,depth-1, -math.inf, math.inf)
            if next_value==v:
            #if action has the same value we also save it
                best_action.append(action)
            if next_value >v:
                best_action = []
                v = next_value
                best_action.append(action)  
        #selects random action from all actions with the same value
        return random.choice(best_action)
    
    if board.whos_turn ==-1:
        v = math.inf
        best_action = [] 
        for action in board.not_full():
            new_board = copy.deepcopy(board)
            new_board.play(action)
            next_value =  max_player(new_board,depth -1, -math.inf, math.inf)
            if next_value==v:
                #if action has the same value we also save it
                best_action.append(action)
            if next_value<v:
                best_action= []
                v = next_value
                best_action.append(action)
        #selects random action from all actions with the same value
        return random.choice(best_action)



