import numpy as np 


class board_state(object):

    def __init__(self):
        b = []
        for i in range(6):
            b.append([0]*7)
        self.current_state = np.array(b) 
        self.turn = 0
        self.whos_turn = 1 
    
    #returns the current state of the board as a vector
    def vect(self):
        v = np.array([0 for i in range(6*7)])
        for i in range(6):
            v[7*i:7*(i+1)] = self.current_state[i] 
        return v 

    #resets the board to play a new game    
    def reset(self):
        b = []
        for i in range(6):
            b.append([0]*7)
        self.current_state = np.array(b) 
        self.turn =0 
        self.whos_turn =1

    def print_board(self):
        for row in self.current_state:
            print(''.join(row))

    #takes a number from 0 to 6 and places to chip in the next free spot in that column (if that column is not already full)
    def play(self,n):
        l= []
        for i in range(6):
            if self.current_state[i,n] ==0:
                l.append(i)
        if l ==[]:
            return False
        if self.whos_turn ==1:
            self.current_state[max(l)][n] = 1 
            self.whos_turn = -1 
            self.turn +=1
        else:
            self.current_state[max(l)][n] = -1 
            self.whos_turn =1
            self.turn += 1 
        return True

   #returns a list that contains the indices of all columns that are not full 
    def not_full(self):
        not_full =[]
        for i in range(7):
            l = []
            for j in range(6):
                if self.current_state[j][i] == 0:
                    l.append(j)
            if l!=[]:
                not_full.append(i)
        return not_full 

    #checks whether there are four chips in a row in any of the admissible directions. p determines for which player the condition is checked.
    def check_win(self,p):
        if p ==1:
            s = 1 
        else:
            s = -1 

        #check horizontal
        def check_h(self,s):
            for i in range(6):
                for j in range(4):
                    c = 0
                    for k in range(4):
                        if self.current_state[i][j+k]== s:
                            c +=1
                    if c==4:
                        return True
            return False

        #check vertical
        def check_v(self, s):
            for i in range(3):
                for j in range(7):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j] == s:
                            c += 1
                        if c ==4:
                            return True
            return False

        #check diagonal
        def check_d(self,s):
            for i in range(3):
                for j in range(4):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j+k] == s:
                            c +=1 
                        if c ==4 :
                            return True
            for i in range(3):
                for j in range(3,7):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j-k] == s:
                            c +=1
                        if c ==4:
                            return True
            return False
            #combind   
        if check_h(self,s) or check_v(self,s) or check_d(self,s):
            return True
        return False

    #used to perform a single step during the learning proceedure. Positions a chip in the n-th column and returns the resulting state as a vector, whether the move was valid and whether the game ended.
    def step(self,n):
        played = self.play(n)
        end = False
        if self.check_win(1) or self.check_win(2) or self.turn==42:
                end = True
        return [self.vect(), played, end]
        
    
   