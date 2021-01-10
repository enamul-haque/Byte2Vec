#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This is to generate fragments of size 4096 bytes from all files from 1000 
stored in govdocs1 directory. 
-------------------------------------------------------------------------------
    Variables:
    
        folder_name = the folder where you have all 1000 folders
        tosave_path = where you save the fragments
        c = fragment size in bytes. default is given 4096, 
            change is to 512 if needed
@inistitute: University of Louisian at Lafayette
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import math
import os
import mmap
import random

"""
find chunk of data from within a block. 
    start: start location of the block
    length: size of the chunk
"""
def get_foreign_chunk(filename, start, length):
    fobj = open(filename, 'r+b')
    m = mmap.mmap(fobj.fileno(), 0)
    return m[start:start+length]

"""
function for splitting the file data into blocks of size n=4096 bytes
"""
def split_by_blocks(infile, n, marker, save_path):

    f = open(infile, 'rb')
    l = len(f.read())
    split_count = math.ceil(l/n) # consider the ceiling of the division as an integer.
    #print("Blocks:", split_count)
    s = np.arange(split_count)
    
    # a folder with mixed types of files to use as part of the last fragment
    loc = "/mixed/" 
    
    with open(infile, 'rb') as f:
        k = 0
        for chunk in iter(lambda: f.read(n), b''):
            i = s[k]
            j = len(chunk)
            
            if (i == split_count-1) and (j < n):
                print("Partial fragments are being createds!")
                # list the files
                filelist = os.listdir(loc)
                # generate a random integer between 1 and 100
                index = random.randint(1,100)
                # select that random file from filelist
                randfile = filelist[index]
                randfile = os.path.join(loc,randfile)
                # get the missing chunk from location j till n
                foreign_chunk = get_foreign_chunk(randfile,j, n-j)
                # append with current chunk
                chunk = chunk + foreign_chunk
                #print("Lenth of the last chunk is:"+str(len(chunk))+"kb")
                print(str(marker)+ "_partial_"+ str(i+1) + "."+ str(infile.rpartition('.')[-1]))
                out_file = str(marker)+"_partial_"+ str(i+1) + "."+ str(infile.rpartition('.')[-1])
            else:
                print(str(marker)+"_full_"+ str(i+1) + "."+ str(infile.rpartition('.')[-1]))
                out_file = str(marker)+"_full_"+ str(i+1) + "."+ str(infile.rpartition('.')[-1])
                    
                out_file = os.path.join(save_path, out_file) 
            
                with open(out_file, 'wb') as ofile:
                        ofile.write(chunk)
            k += 1

"""
function for splitting the file based on specific marker other than newline (\n)
this function is not used, its for future improvement.
"""
def split_by_marker(f, marker = "-MARKER-", block_size = 4096):
    print("Start")
    current = ''
    while True:
        block = f.read(block_size)
        if not block: # end-of-file
            yield current
            return
        current += block
        while True:
            markerpos = current.find(marker)
            if markerpos < 0:
                break
            yield current[:markerpos]
            current = current[markerpos + len(marker):]

    print(current)

if __name__ == "__main__":
    
    # run it on local or server?
    local = 0 # 0=online, 1=local
    
    if local==1:
        folder_name = './'
        tosave_path = './dump/'
    else:
        folder_name = '/govdocs_files_unzipped/' # location of unzipped files from govdocs1
        tosave_path = '/fragment_data/' # location to save fragments
    
    # this is to generate all 1000 folder names
    folders = list()
    for i in range(1000):
        # 000 directory has problem in files, start from 001
        if i>=0 and i< 10:
            #print(str('00')+str(i))
            folders.append(str('00') + str(i))
        if i >= 10 and i < 100:
            #print(str('0')+str(i))
            folders.append(str('0') + str(i))
        if i >= 100 and i< 1000:
            #print(i)
            folders.append(str(i))
        
                
    locs = folders
    c = 4096
    save_path = tosave_path
    for i in range(len(locs)):
        os.system ("mkdir "+ str(save_path) + str(locs[i])) # create a folder
        path = folder_name + str(locs[i])
        save_path = str(save_path) + str(locs[i]) + str('/') # directory to save individual fragments of a folder
        for f in os.listdir(path):
            file = os.path.join(path, f)        
            marker = os.path.splitext(f)[-2]
            extension = os.path.splitext(file)[-1]
            fileType = extension.upper()
            split_by_blocks(file,c, marker, save_path)        
        save_path = tosave_path # reset the save path        
    print("Program ended running..")