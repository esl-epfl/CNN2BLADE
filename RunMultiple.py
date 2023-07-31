# --------------------------------------------------------------------------------
# -- Title       : Analysis of convolutional layers - batch analysis
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : RunMultiple.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 9 2021
# --------------------------------------------------------------------------------
# -- Module description : File to perform the full analysis of the convolutional
# -- layers of all the models in 'models'
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
from RunSingle import RunSingleLayer
import csv

# --------------------------------------------------------------------------------
# import the variables CNNname and the convolutional layers from the selected model.
# --------------------------------------------------------------------------------

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
        from models.custom   import CNNname, conv_layer_list
    else:
        input("enter a valid CNN")
        valid_model = 1
print("CNN name:", CNNname)

# --------------------------------------------------------------------------------
# Naive: maximum parallelization by maximizing data transfer
# Non-Naive: Minimum data transfer, resulting in some cases to limited parallelization. 
#
# Cycles MAC: cycles on the accumulations  
#           1 - TETC architecture 
#           2 - CODES architecture (MACL, MACH) 
# --------------------------------------------------------------------------------
naive_or_not_list = [0,1]
cycles_MAC_list = [1,2]

for naive_or_not in naive_or_not_list:
	for cycles_MAC in cycles_MAC_list:
    # create the csv writer
		if naive_or_not == 1:
			f = open('Results/' + str(model) + "Naive_"+ str(cycles_MAC) +"cycleMAC.csv", 'w', newline='')
		else: 
			f = open('Results/' + str(model) + "NonNaive_"+ str(cycles_MAC) +"cycleMAC.csv", 'w', newline='')
		writer = csv.writer(f)


		print_en = 0  # print enable: 0 to avoid any print, 1 to print analysis  
		word_size = 16 #bit-width of word memory (different from the IMO quantization!)
		words_bus_vector = [1] #amount of words transfered in a bus per cycle


		# this for loop generates the size of the subarrays for the batch analysis
		# from 64 to 8192
		subarray_size_words_vector = []
		potencies = list(range(6,14))
		for pw in potencies:
			subarray_size_words_vector.append(2**pw)


		# this loop performs the layers based analysis considering different sizes of memory. 
		for words_bus in words_bus_vector: 
			number_of_layer = 0
			for conv_layer_el in conv_layer_list:
				for subarray_size_words in subarray_size_words_vector:
                    # the function RunSingleLayer is defined in RunSingle.py
					RSL = RunSingleLayer(conv_layer_el,subarray_size_words,word_size, cycles_MAC,words_bus, print_en,naive_or_not)
					row = [number_of_layer] + RSL
					print(row)
					writer.writerow(row)
				number_of_layer += 1
		f.close()
