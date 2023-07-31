# --------------------------------------------------------------------------------
# -- Title       : Analysis of convolutional layers
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : RunSingle.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 9 2021
# --------------------------------------------------------------------------------
# -- Module description : This file contains a function that structure the call 
# -- of the conv tilling. It is also used by RunMultiple.py
#
#
# > 'python RunMultiple.py X' where X is: 
# 
# -----> 1 - GoogleNet
# -----> 2 - Mobilenet
# -----> 3 - Resnext
# -----> 4 - Alexnet
# -----> 5 - VGG16
# -----> 6 - Resnet8
# -----> 7 - Custom
# --------------------------------------------------------------------------------

import sys, math
from Conv_Functions import top_level_layer_tilling
from Conv_Class import conv_layer

model = int(sys.argv[1])
valid_model = 1 
while valid_model == 1:
    valid_model = 0
    if model == 1:
        from models.googlenet import CNNname, conv_layer_list
    elif model == 2:
        from models.mobilenet import CNNname, conv_layer_list
    elif model == 3:
        from models.resnext   import CNNname, conv_layer_list
    elif model == 4:
        from models.alexnet   import CNNname, conv_layer_list
    elif model== 5:
        from models.vgg       import CNNname, conv_layer_list
    elif model== 6:
        from models.resnet8   import CNNname, conv_layer_list
    elif model== 7:
        from models.custom     import CNNname, conv_layer_list
    else:
        input("enter a valid CNN")
        valid_model = 1
print("CNN name:", CNNname)

# --------------------------------------------------------------------------------
# Function RunSingleLayers: Perfoms the analysis of a single convolutional layer 
# this function is also used in the RunMultiple.py. 
# 
# Inputs: 
#  conv_layer_el: conv_layer element from the conv_layer_list from the CNN models
#  words16b_sub: Size of subarray, in number of words
#  bitwidth_memory: bit-width of word memory (different from the IMO quantization!)
#  cycles_MAC: 1 or 2, depending on the architecture considered
#  words_bus: amount of words transfered in a bus per cycle 
#  print_en: 0 to avoid any print, 1 to print analysis   
#  naive_or_not: Naive: maximum parallelization by maximizing data transfer
#       Non-Naive: Minimum data transfer, resulting in limited parallelization.
# --------------------------------------------------------------------------------
def RunSingleLayer(_layer_,words16b_sub,bitwidth_memory, cycles_MAC,words_bus, print_en, naive_or_not):


    # top_level_layer_tilling (defined in Conv_Functions.py) 
    # s = [input_block, output_block, total_words, rounds]
    #ops =[Multiplications, stream in, stream out]
    s,ops= top_level_layer_tilling(_layer_, print_en, words16b_sub, naive_or_not)

    #minimum data transfers, it considers the ideal case where there isnt data redundacy
    min_dtout = _layer_.get_dtout()*cycles_MAC
    min_dtin = _layer_.get_dtin()
    min_dt = min_dtout + min_dtin

    #data transfer considering the mapping 
    total_dtout = (ops[2]*cycles_MAC)*s[3]
    total_dtin = ops[1] *s[3]
    total_dt = total_dtout + total_dtin

    data_transfer_overhead = "{:.2f}".format(total_dt/min_dt)

    #ratio of each block and the layer, in a given dimension
    Out_Y_ratio = (_layer_.get_output()[1]/s[1][1])
    Out_X_ratio = (_layer_.get_output()[0]/s[1][0])
    In_Z_ratio = (_layer_.input[2]/s[0][2])

    #minimum to parallelize: tilling of output are only considered in the X and Y ratio 
    min_2_MMSM = Out_X_ratio*Out_Y_ratio

    if print_en == 1:
        print("\n")
        print("Weight quantization:", _layer_.weightQ)
        print("Activations quantization:", _layer_.activQ)
        print("\n")
        print("Ideal/ total DT in:", min_dtin,", ", total_dtin)
        print("Ideal/ total DT out:", min_dtout,", ", total_dtout)
        print("Total DT:",total_dt,"Minimum DT:",min_dt,"Ratio:", data_transfer_overhead )
        print("\n")



    return([words16b_sub,_layer_.type,ops[0], total_dt, [int(min_2_MMSM), int(min_2_MMSM*In_Z_ratio)], data_transfer_overhead, _layer_.weightQ, _layer_.activQ])


######################
# --main code 
######################

print_en = 1

words16b_sub = 512
bitwidth_memory = 16
words_bus = 1
cycles_MAC = 2
naive_or_not = 0

for conv_layer_el in conv_layer_list:
        _layer_ = conv_layer(*conv_layer_el)	
        a = RunSingleLayer(_layer_,words16b_sub,bitwidth_memory, cycles_MAC,words_bus, print_en,naive_or_not)
        #print(a)