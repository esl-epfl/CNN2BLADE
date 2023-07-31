# --------------------------------------------------------------------------------
# -- Title       : convolutional layer description 
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : custom.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 4 2021
# --------------------------------------------------------------------------------
# -- Module description : This file, similar to all from this folder, provides the
# -- the models for CNNs 
# --------------------------------------------------------------------------------


CNNname = "custom"

# CONV = [(A,A,B),C,D,E,F,G,H]
#         A -> input layer width and lenght 
#         B -> input layer depth
#         C -> output layer depth 
#         D -> kernel width and lenght
#         E -> stirde
#         F -> "Conv" or "SeparableConv"
#         G -> Weights quantization (BO)
#         H -> Activation Quantization (IMO)
conv_layer_list = [
[(32,32,30),64, 5, 1, "Conv", 8, 16],
]


# FC = [input, output, weights quantization (IMO), activation (BO) ]
fc_layer_list = [
[1024,100,16,5]
]
