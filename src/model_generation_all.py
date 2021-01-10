#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@inistitute: University of Louisian at Lafayette
Byte2Vec modeling of the raw frequency distribution for the file fragments
if the vocabulary turns out to be empty, then no model will be generated
-------------------------------------------------------------------------------
    Variables:
    
        path = sample fragments location on which the model will be built
        sizes = vector size starting from 5 to 100 with 5 interval
        output = byte2vec models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
from nltk.tokenize import word_tokenize
import gensim
import time
import numpy as np

if __name__ == "__main__":
    
    local = 0
    if local==1:
        path = './dump/000/'
    else:
        path = './sampled_data/'
        
    start = time.clock()
    #sizes = [20,50,100,150,200] 
    sizes = list(np.arange(5,105,5)) # vector length
    no_vocab_cnt = 0 # finds if any model type is missed or not
    f = open('model_gen_time_stat.txt','a')
    f.write("vector_size"+"\t"+"model_gen_time"+"\n")

    for i in range(len(sizes)):
        s = time.clock()
        size = sizes[i]
        vocab = []
        count = 0
        
        for file in os.listdir(path):
            count = count + 1    
            current = os.path.join(path, file)
            extension = os.path.splitext(current)[-1]
            fileType = extension[1:].lower()
            cur_file = open(current, "rb")
            data = bytearray(cur_file.read())
            
            data_temp = ''
            for i in range(len(data)):
                data_temp = data_temp + " " + str(int(data[i]))
            data_temp = data_temp.strip()
            data_temp = word_tokenize(data_temp)
            vocab.append(data_temp)
            
            if (count % 200) == 0:
                print("Fragment", count, "is processed.\n************************")        
            #model = gensim.models.Word2Vec(vocab, min_count=1)

        model = gensim.models.Word2Vec(vocab,size=size,window=5,min_count=1,workers=4)
        model.wv.save_word2vec_format("byte2vec_model_vecsize_"+str(size))
        #model.save("test_w2v_forensics_model")
        e = time.clock()
        el = e-s
        f.write(str(size)+"\t"+str(el)+"\n")
        print("Done for vecotr length: ", str(size))
    end = time.clock()
    print("Voila! finished building and saving model for fragments!\n")
    if (no_vocab_cnt == 0):
        print("No missed vocabulary\n")
    else:
        print(no_vocab_cnt, " missed vocabualry, please check!\n")    
    print("Time elapsed:", format(round((end-start)/3600,4)), "hours \nTotal files processed:", count)
    #model = gensim.models.Word2Vec(vocab,size=size,window=5,min_count=1,workers=4)
    #model.save_word2vec_format("test_model")
    #new_model = gensim.models.word2vec.Word2Vec.load_word2vec_format("test_model")