from CFalphaBeta import alphaBeta
import random


#define a class that represents a player without knowledge, i.e. randomly selects a (possible) action
class randomPlayer():

    def __init__(self):
        pass

    def play(self,board):
        action = random.choice(board.not_full())
        return board.step(action)

#defines a class that representats a player that makes decisions based on alpha-beta pruning
class alphaBetaPlayer:
    
    def __init__(self, deepth):
        self.deepth = deepth

    def play(self,board):
        return board.step(alphaBeta(board, self.deepth))

