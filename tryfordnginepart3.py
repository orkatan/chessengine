import chess
import chess.pgn
import chess.polyglot
import cProfile
import json
from keras import models
import tensorflow as tf
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plot
import numpy as np
from keras import regularizers
from keras import Sequential
from keras.utils import Sequence
from keras.layers.core import Dense
import keras
from keras import callbacks
import random
import copy
from keras.optimizers import adamw_experimental
from keras import activations
import numpy as np
import sqlite3

# Establish a connection to the Database and create
# a connection object
keeptrake=0
def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b
m=np.zeros((1))

boardkeep=np.zeros((1,768))
scoretest=np.zeros((1))
kop=np.zeros((1,768))
l=np.zeros((1,768))
y_valpart=np.zeros((1))
x_valpart=np.zeros((1,768))
k=int(input("enter the number of times that he trained for"))

if(k==0):
    board_model1 = Sequential()

    board_model1.add(Dense(1000 , activation='relu',input_shape=(768,)))
    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    pgn = open('lichess_db_standard_rated_2017-01.pgn')

    board_model1.add(Dense(1000 , activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
    board_model1.add(Dense(1 , kernel_regularizer=regularizers.l1(0.0001)))
if(k>0):
    json_file = open('board_model431.json', 'r')
    board_json = json_file.read()
    json_file.close()
    board_model1 = keras.models.model_from_json(board_json)
    board_model1.load_weights('reinforced_model4531.h5')
board_model1.compile(optimizer='adam', loss='mae',metrics=["mse"])
if(k>0):
    pgn = open('lichess_db_standard_rated_2017-01.pgn',buffering=k)



def counbitts(whatarry):
    counts = (bytes(bin(whatarry).count("1") for x in range(1)))
    return(counts[0])

#if(k>0):
    #for j in range(0,k):
    #first_game = chess.pgn.read_game(pgn,k)
        #if(j%1000==0):
            #print(j)
for g in range(0,3000000):
    
    first_game = chess.pgn.read_game(pgn)
    turn=1
    
    if(g%500==0):
        print(g)
    if g%15000==0 and g>1:
        #board_model1.compile(optimizer='adam', loss='mse', metrics=["mae"])

        board_json = board_model1.to_json()
        with open('board_model431.json', 'w') as json_file:
            json_file.write(board_json)
        board_model1.save('reinforced_model4531.h5')

    if((len(kop)>5000) and g>0):
        print(g)
        j=10
        #board_model1.compile(optimizer='adam', loss='MeanAbsoluteError')
        history=board_model1.fit(kop[1:], m[1:], epochs=3,batch_size=1024, verbose=1,validation_data=(x_valpart,y_valpart),shuffle=True)
        print("hello")
        #plot.plot(history.history['mse'])
        #plot.plot(history.history['val_mse'])
        #plot.title('model accuracy')
        #plot.ylabel('mse')
        #plot.xlabel('epoch')
        #plot.legend(['train', 'validation'], loc='upper left')
        #plot.show()
        #plot.plot(history.history['loss'])
        #plot.plot(history.history['val_loss'])
        #plot.title('model loss')
        #plot.ylabel('loss')
        #plot.xlabel('epoch')
        #plot.legend(['train', 'validation'], loc='upper left')
        #plot.show()
        kop=np.zeros((1,768))
        m=np.zeros((1))
        if(len(y_valpart)>=1500):
            y_valpart=np.zeros((1))
            x_valpart=np.zeros((1,768))
        print(keeptrake)
    for move in first_game.mainline_moves():
        first_game=first_game.next()
        #if(first_game.eval()!=None and (not first_game.eval().is_mate()) and turn<40):
        if(first_game.eval()!=None and turn<40 and (not first_game.eval().is_mate())):

                #print(first_game.board())
                #print(first_game.eval())
                if(turn%2==1):
                    board=first_game.board()
                    #print(board)
                    #print(first_game.eval())
                    black, white = board.occupied_co
                    bitboards = np.array([black & board.pawns, black & board.knights, black & board.bishops, black & board.rooks, black & board.queens, black & board.kings, white & board.pawns, white & board.knights, white & board.bishops, white & board.rooks, white & board.queens, white & board.kings, ], dtype=np.uint64)

                    #print(kop.shape)
                    #print(boardwhoread.shape)
                    if(first_game.eval().is_mate()):
                        y=(((first_game.eval().white())).mate())
                        score=1500

                        if(y<0):
                            score=-1500

                    if(not first_game.eval().is_mate()):
                        score=first_game.eval().white().score()
                    score=float(score/100)
                    #print(score)
                if(turn%2==0):
                    board=first_game.board().mirror()
                    #print(board)
                    #print(first_game.eval())
                    black, white = board.occupied_co
                    bitboards = np.array([black & board.pawns,black & board.knights,black & board.bishops,black & board.rooks,black & board.queens,black & board.kings,white & board.pawns,white & board.knights,white & board.bishops,white & board.rooks,white & board.queens,white & board.kings,], dtype=np.uint64)
                    if(first_game.eval().is_mate()):
                        y=(((first_game.eval().black())).mate())
                        score=1500
                        if(y<0):
                            score=-1500
                    if(not first_game.eval().is_mate()):
                        score=first_game.eval().black().score()
                    score=float(score/100)
                    #print(score)
                if(score>15):
                    score=15
                if(-15>score):
                    score=-15
                score=float(score)
                score=score*-1
                #print(score)
                #if(first_game.turn()==chess.WHITE):
                #    print(first_game.eval().white())
                #if(first_game.turn()==chess.BLACK):
                #    print(first_game.eval().black())
                #print("")
                #boardwhoread=np.concatenate((boardwhoread),axis=0)
                #boardwhoread=np.concatenate((boardwhoread),axis=0)
                #bitboards=np.reshape(bitboards,(1,12))


                bitboards = bitboards_to_array(bitboards)
                bitboards=np.reshape(bitboards,(1,768))
                if(len(y_valpart)==1400):
                    print(score)
                    print(board_model1.predict_on_batch(bitboards))

                    print(first_game.board())
                if(g%5==0 and len(y_valpart)<1500):
                    x_valpart=np.vstack((x_valpart,bitboards))
                    y_valpart=np.vstack((y_valpart,score))
                if((not g%5==0) or len(y_valpart)>=1500):
                    m=np.vstack((m,score))
                    kop=np.vstack((kop,bitboards))
                turn=turn+1
                keeptrake=keeptrake+score

#with open('savetohere', 'wb') as f:
#    np.save(f, boardkeep)

board_json = board_model1.to_json()
with open('board_model451.json', 'w') as json_file:
	json_file.write(board_json)
board_model1.save('reinforced_model4531.h5')
