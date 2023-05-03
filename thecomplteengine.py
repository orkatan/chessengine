import chess.polyglot
#import matplotlib.pyplot as plot
import keras
#from keras.optimizers import adamw_experimental
import numpy as np
import copy

json_file = open('board_model4.json', 'r')
board_json = json_file.read()
json_file.close()
reinforced_model = keras.models.model_from_json(board_json)
reinforced_model.load_weights('../reinforced_model45.h5')
boardtostart=chess.Board()
def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)
def anylyz(current_board, time3=1, minvalue=0, v=1):
    #print(boards)
    
    if(time3==2):
        list_of_moves = current_board.legal_moves
        number_of_possible_moves_for_next_turns = []
        two_turns_possible_boards = np.zeros((1, 768))
        if current_board.is_checkmate():
            return 100
        if current_board.is_stalemate():
            return 0
        for current_move in list_of_moves:

            current_board.push(current_move)
            next_turn_possible_boards = anylyz(current_board, 1)
            #p=copy.deepcopy(next_turn_possible_boards)
            number_of_possible_moves_for_next_turns.append(len(next_turn_possible_boards))
            two_turns_possible_boards = np.vstack((two_turns_possible_boards, next_turn_possible_boards))

            current_board.pop()
        evaluation_for_each_board=(reinforced_model.predict(x=two_turns_possible_boards, verbose=0))
        m=[]
        for d in number_of_possible_moves_for_next_turns:

            m.append((min(evaluation_for_each_board[:d])))
            evaluation_for_each_board = evaluation_for_each_board[d:]

        return (max(m))
    #if time3==1:
        # = np.zeros((1, 768))
        #istofmoves1=current_board.legal_moves
        #or movemnt in listofmoves1:
        #   current_board.push(movemnt)
        #   l = anylyz(current_board, 0)
        #   #print(l)
        #   k = copy.deepcopy(l)
        #   r = np.vstack((r, k))
        #   current_board.pop()
        #evaluation_for_each_board=(reinforced_model.predict(x=o,verbose=0))

        #print(evaluation_for_each_board)
        #print(np.max(evaluation_for_each_board))
        #print(boards)
        #for y in evaluation_for_each_board:
        #    print(y)
        #    print(listofmoves1)
        #    print(count)
        #    count=count+1
        #print(evaluation_for_each_board)
        #return(min(evaluation_for_each_board))
        #return r
    
    #if(boards.is_checkmate()):
    #    return(100)
    #if(boards.is_stalemate()):
    #    return(0)
    board = current_board
    #print(board)
    black, white = board.occupied_co
    bitboards = np.array([black & board.pawns,black & board.knights,black & board.bishops,black & board.rooks,black & board.queens,black & board.kings,white & board.pawns,white & board.knights,white & board.bishops,white & board.rooks,white & board.queens,white & board.kings], dtype=np.uint64)
    boardwhoread=bitboards_to_array(bitboards)
    boardwhoread=np.concatenate((boardwhoread),axis=0)
    boardwhoread=np.concatenate((boardwhoread),axis=0)
    #evaluation_for_each_board=(reinforced_model.predict(x=np.array([boardwhoread],),verbose=0))
    #print(evaluation_for_each_board)
    #print(board)
    return(boardwhoread)


def moredeathnopower(boarding):
    minrev = 100
    listofmoves = boarding.legal_moves
    for movemnt in listofmoves:
        boarding.push(movemnt)
        j = anylyz(boarding,2,minrev+0.5)
        minrev = min(j, minrev)
        boarding.pop()
    print(minrev)
    return minrev
game1=True
well=-10
turn=1
minrev=-100
#booard=chess.Board()
#for i in range(0,10):
#    listofmoves=booard.legal_moves
#    count=0
#    count1=0
#    for h in listofmoves:
#        print(h)
#        print(count1+1)
#        count1=count1+1
#
#    k = int(input("enter move"))
#    for y in listofmoves:
#        count=count+1
#        if(count==k):
#            move=y
#    booard.push(move)
#    board = booard
#    black, white = board.occupied_co
#    bitboards = np.array(
#        [black & board.pawns, black & board.knights, black & board.bishops, black & board.rooks, black & board.queens,
#         black & board.kings, white & board.pawns, white & board.knights, white & board.bishops, white & board.rooks,
#         white & board.queens, white & board.kings, ], dtype=np.uint64)
#    boardwhoread = bitboards_to_array(bitboards)
#    boardwhoread = np.concatenate((boardwhoread), axis=0)
#    boardwhoread = np.concatenate((boardwhoread), axis=0)
#    maxt=(reinforced_model.predict(x=np.array([boardwhoread],),verbose=0))
#    print(maxt)
#    print(booard)

while(game1==True):

    if(turn%2==1):
        minrev=100
        #print(boardtostart.legal_moves())
        #move=input("write yor move")
        listofmoves=boardtostart.legal_moves

        count=0
        j=1
        for movemnt in listofmoves:
            count=count+1
            boardtostart.push(movemnt)
            j=anylyz(boardtostart,2,minrev+0.5)
            minrev = min(minrev,j)
            if(minrev==j):
                move=movemnt
            boardtostart.pop()
    if(turn%2==0):
        minrev=-100
        count=0
        listofmoves=boardtostart.legal_moves
        #count=0
        #count1=0
        #for h in listofmoves:

        #    print(h)
        #    print(count1+1)
        #    count1=count1+1
        #k = int(input("enter move"))
        #for y in listofmoves:
        #    count=count+1
        #    if(count==k):
        #        move=y
        for movemnt in listofmoves:
            count=count+1
            print(count)
            boardtostart.push(movemnt)
            print(movemnt)
            j=moredeathnopower(boardtostart)
            minrev=max(minrev, j)
            if(minrev==j):
                move=movemnt
            boardtostart.pop()
            print(minrev)
    boardtostart.push(move)
    turn=turn+1
    print(boardtostart)
    if(boardtostart.is_checkmate()):
        break
