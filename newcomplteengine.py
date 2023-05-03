import chess.polyglot
# import matplotlib.pyplot as plot
import keras
# from keras.optimizers import adamw_experimental
import numpy as np
import copy
import cProfile
import re
import tensorflow as tf
import time

print(tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()

json_file = open('board_model431.json', 'r')
board_json = json_file.read()
json_file.close()
reinforced_model = keras.models.model_from_json(board_json)
reinforced_model.load_weights('reinforced_model4531.h5')

boardtostart = chess.Board()

def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    #return b.reshape(-1, 8, 8)
    return b
def formap(argsi,yu):
    return(evalfunc[argsi].sum()+yu)
def thisfun(nums):
    return((nums[0]+(nums[1]+nums[2])*3+nums[3]*5+nums[4]*9)*100)
def simplfunc(pos,mey1):
    u=list(map(np.nonzero,pos))
    scom=list(map(thisfun,mey1))
    o=list(map(formap,u,scom))
    return(o)
def anylyz(current_board, isreturn, whosturn, death):
    if (isreturn == 1):
        if(whosturn==1):
           current_board=current_board.mirror()

        list_of_moves = current_board.legal_moves
        con=[]
        ne=[]
        two_turns_possible_boards = np.zeros((1, 768),dtype='uint64')
        if current_board.is_checkmate():
            two_turns_possible_boards = np.vstack((two_turns_possible_boards, anylyz(current_board, 0, whosturn, 0)[0]))
            return (two_turns_possible_boards,2)
        for tovit in list_of_moves:
            current_move = chess.Move.from_uci(str(tovit))
            current_board.push(current_move)
            next_turn_possible_boards,nom,e= anylyz(current_board, 0,whosturn,death-1)
            #number_of_possible_moves_for_next_turns.append(len(next_turn_possible_boards))
            con.append(e)
            ne.append(nom)
            two_turns_possible_boards = np.vstack((two_turns_possible_boards,next_turn_possible_boards))
            current_board.pop()
        #for c in two_turns_possible_boards:
        #    evovole.append(bitboards_to_array(c))
        #evovole=np.reshape(evovole,(len(keepmetri),768))
        #evaluation_for_each_board =(((reinforced_model.predict_on_batch(evovole))))
        #count=0
        #for point in keepmetri:
        #    evaluation_for_each_board[count] = evaluation_for_each_board[count]/2+ float(point/2)
        #    count = count + 1
        #m = []
        #for d in number_of_possible_moves_for_next_turns:
        #    m.append((max(evaluation_for_each_board[:d])))
        #    evaluation_for_each_board = evaluation_for_each_board[d:]
        #if(whosturn==1):
        #    current_board=current_board.mirror()
        ###return (two_turns_possible_boards,keepmetri,number_of_possible_moves_for_next_turns)
        #return(min(m))
        if(whosturn==1):
            current_board.mirror()
        two_turns_possible_boards=two_turns_possible_boards[1:]
        two_turns_possible_boards=two_turns_possible_boards[np.argmin(ne)]
        con=con[np.argmin(ne)]
        return(two_turns_possible_boards,con) #number_of_possible_moves_for_next_turns)
    if death >0:
       r=[]
       listofmoves1 = current_board.legal_moves
       con=[]
       for povit in listofmoves1:
            movemnt = chess.Move.from_uci(str(povit))
            current_board.push(movemnt)
            l,e = anylyz(current_board, 0, whosturn, death - 1)
            con.append(e)
            r.append(l)
            current_board.pop()

       evaluation_for_each_boards=np.zeros((len(r),768))
       count=0
       for h in r:
            evaluation_for_each_boards[count]=((bitboards_to_array(h)))
            count=count+1

       if current_board.is_checkmate():
           r=(anylyz(current_board, 0, whosturn, 0)[0])
           coun = []
           count=0
           for y in r[:5]:
               coun.append(y.bit_count() - r[count + 6].bit_count())
               count = count + 1
           r=bitboards_to_array(r)
           return (r,-100000,coun)
        # print(evaluation_for_each_board)
        # print(np.max(evaluation_for_each_board))
        # print(boards)
        # for y in evaluation_for_each_board:
        #    print(y)
        #    print(listofmoves1)
        #    print(count)
        #    count=count+1
        # print(evaluation_for_each_board)
        # return(min(evaluation_for_each_board))
       #simnum=[]
       #count=0

       #p = cProfile.Profile()
       #p.runcall(simplfunc,evaluation_for_each_boards,con)
       #p.print_stats(sort="tottime")

       simnum=simplfunc(evaluation_for_each_boards,con)
       #for q in evaluation_for_each_boards:
           #simnum.append(simplfunc(q,con[count]))
           #count=count+1
       b=np.argmax(simnum)
       evaluation_for_each_boards=evaluation_for_each_boards[b]
       con=con[b]
       #print(current_board)
       #print(simplfunc(bitboards_to_array(anylyz(current_board,0,whosturn,0)[0])))
       return (evaluation_for_each_boards,max(simnum),con)

    # if(boards.is_checkmate()):
    #    return(100)
    # if(boards.is_stalemate()):
    #    return(0)
    board = current_board



    black, white = board.occupied_co
    #if (whosturn == 1):
        #board = current_board.mirror()
        #black, white = board.occupied_co
    bitboards = np.array(
        [black & board.pawns, black & board.knights, black & board.bishops, black & board.rooks, black & board.queens,
         black & board.kings, white & board.pawns, white & board.knights, white & board.bishops, white & board.rooks,
         white & board.queens, white & board.kings], dtype=np.uint64)
    #bit=np.array([black],dtype=np.uint64)
    #boardwhoread = bitboards_to_array(bitboards)
    #bit=bitboards_to_array(bit)
    #thingtokeephere=(getefficent(boardwhoread,bit))
    #boardwhoread = np.concatenate((boardwhoread), axis=0)
    #boardwhoread = np.concatenate((boardwhoread), axis=0)
    bitboards.reshape((1,12))
    #evaluation_for_each_board=(reinforced_model.predict_on_batch(boardwhoread))
    #if(evaluation_for_each_board>3 or evaluation_for_each_board<-3):
        #print(evaluation_for_each_board)
        #print(board)
    coun=[]
    count=0
    for y in bitboards[:5]:
        coun.append(y.bit_count()-bitboards[count+6].bit_count())
        count = count + 1
    if current_board.is_checkmate():
        return (bitboards,coun)
    #return (boardwhoread,-thingtokeephere)
    return(bitboards,coun)





game1 = True
ell = -10
turn = 1
minrev = -100
booard=chess.Board()



game1=True




def gamewithself():
    turn=1
    while (game1 == True):
        start = time.time()

        material=[]
        keepingboardnumbers=[]
        keepuppersubboard=[]
        lost=[]
        material=[]
        finalboardssave=np.zeros((1,768),dtype='uint64')
        listofmoves = (boardtostart.legal_moves)
        for movit in listofmoves:
            #count = count + 1
           movemnt=chess.Move.from_uci(str(movit))
           boardtostart.push(movemnt)
            #j = anylyz(boardtostart, 1, turn % 2, 2)
            ##print(movemnt)
            ##print(j)
#
            #minrev = max(j, minrev)

            #if (minrev == j):
            #    move = movemnt
            #boardtostart.pop()
           boardstoread,mot = anylyz(boardtostart, 1, turn % 2,2)
           finalboardssave=np.vstack((finalboardssave,boardstoread))
           #keepfinameti=np.vstack((keepfinameti,materialtokeep))
           #keepingboardnumbers.append(numberofsubboards)
           #keepuppersubboard.append(len(finalboardssave))
           material.append(mot)
           lost.append(movemnt)
           boardtostart.pop()

        finalboardssave=finalboardssave[1:]
        #finalboardssave=np.reshape(finalboardssave,(len(finalboardssave),768))
        evovole = []
        for c in finalboardssave:
            evovole.append(c)
        evovole=np.array(evovole)
        mat=[]

        for count in range(0,len(evovole)):
            mat.append(0)
            counu=0
            for i in material[count]:
                y=i*thisdict[counu]
                mat[count]=mat[count]+y
                counu=counu+1

        evaluation_for_each_board=reinforced_model.predict(x=evovole, verbose=1, batch_size=len(evovole))
        cout=0
        for v in evaluation_for_each_board:
            evaluation_for_each_board[cout]=float(v*1/2)+((float(mat[cout])*(1/2)))
            cout=cout+1
        j=evaluation_for_each_board
        #count=0
        #for k in keepuppersubboard:
        #   m = []
        #   for d in keepingboardnumbers[count]:
        #       m.append(np.max(evaluation_for_each_board[:d]))
        #       evaluation_for_each_board = evaluation_for_each_board[d:]
        #   #keepingboardnumbers=keepingboardnumbers[k:]
        #   count=count+1
        #   j.append(min(m))
        for i in range(0,len(j)):
            print(mat[i])
            print(j[i])
            print(lost[i])
        move = chess.Move.from_uci(str(lost[np.argmax(j)]))
        boardtostart.push(move)
        turn = turn + 1
        print(boardtostart)
        end = time.time()
        print(end - start)
        if (boardtostart.is_checkmate()):
         break
        if(turn==10):
         return()
evalfunc= [0,  0,  0,  0,  0,  0,  0,  0,
50, 50, 50, 50, 50, 50, 50, 50,
10, 10, 20, 30, 30, 20, 10, 10,
 5,  5, 10, 25, 25, 10,  5,  5,
 0,  0,  0, 20, 20,  0,  0,  0,
 5, -5,-10,  0,  0,-10, -5,  5,
 5, 10, 10,-20,-20, 10, 10,  5,
 0,  0,  0,  0,  0,  0,  0,  0,
-50,-40,-30,-30,-30,-30,-40,-50,
-40,-20,  0,  0,  0,  0,-20,-40,
-30,  0, 10, 15, 15, 10,  0,-30,
-30,  5, 15, 20, 20, 15,  5,-30,
-30,  0, 15, 20, 20, 15,  0,-30,
-30,  5, 10, 15, 15, 10,  5,-30,
-40,-20,  0,  5,  5,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50,
-20,-10,-10,-10,-10,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5, 10, 10,  5,  0,-10,
-10,  5,  5, 10, 10,  5,  5,-10,
-10,  0, 10, 10, 10, 10,  0,-10,
-10, 10, 10, 10, 10, 10, 10,-10,
-10,  5,  0,  0,  0,  0,  5,-10,
-20,-10,-10,-10,-10,-10,-10,-20,
0, 0, 0, 0, 0, 0, 0, 0,
5, 10, 10, 10, 10, 10, 10, 5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
0, 0, 0, 5, 5, 0, 0, 0,
-20,-10,-10, -5, -5,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  0,  5,  5,  5,  5,  0,-10,
 -5,  0,  5,  5,  5,  5,  0, -5,
  0,  0,  5,  5,  5,  5,  0, -5,
-10,  5,  5,  5,  5,  5,  0,-10,
-10,  0,  5,  0,  0,  0,  0,-10,
-20,-10,-10, -5, -5,-10,-10,-20,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-20,-30,-30,-40,-40,-30,-30,-20,
-10,-20,-20,-20,-20,-20,-20,-10,
 20, 20,  0,  0,  0,  0, 20, 20,
 20, 30, 10,  0,  0, 10, 30, 20]
evalfunc2=[0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40, -30, 5, 10, 15, 15, 10, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 10, 15, 15, 10, 0, -30, -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50, -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10, 10, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20, 0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 5, 0, 0, 0, 0, -10, -10, 5, 5, 5, 5, 5, 0, -10, 0, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5, -10, 0, 5, 5, 5, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20, 20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20, -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30]
evalfunc2=np.dot(-1,evalfunc2)
evalfunc=np.array(evalfunc)
evalfunc=np.append(evalfunc,evalfunc2)

thisdict={0:1,1:3,2:3,3:5,4:9}
thisdict2=np.array([1,3,3,5,9])

gamewithself()
boardtostart=chess.Board()

p = cProfile.Profile()
p.runcall(gamewithself)
p.print_stats(sort="tottime")
#gamewithself()