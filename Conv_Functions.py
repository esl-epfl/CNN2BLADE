# --------------------------------------------------------------------------------
# -- Title       : convolutional layer functions 
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : Conv_Functions.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 1 2021
# --------------------------------------------------------------------------------
# -- Module description : all the function to tile and map CNN in subarrays. 
# --------------------------------------------------------------------------------


import math
from Conv_Class import conv_layer
import copy

#verify if the tilling proposed may be distributed uniformely for the subarrays
def check_ratios(x,y,z):
	if x%1!=0 or y%1!=0 or z%1!=0:
		print("\n#####################       LAYER TILLING WENT WRONG       #####################\n")


def layer_tilling(_layer_, words16b_sub):

	#local copies of layer structure
	layer = [_layer_.input[0],_layer_.input[1],_layer_.input[2]]
	kernel = _layer_.kernel

 	#output_layer begins with the original size of the layer, and depending on the size of
 	#the subarray, the output is reduced until it fits 
	output_layer = [_layer_.get_output()[0],_layer_.get_output()[1],_layer_.get_output()[2]]
	total_output_words = output_layer[0]*output_layer[1]*output_layer[2]

	layer_done = 0
	flag_layer_modif = 0

	input_depth_reduction_factor = 1
	output_dimensions_reduction_factor = 1 

	layer_original_depth = layer[2]
	input_layer_original_dimension = layer[0]
	output_layer_original_dimension = output_layer[0]


	#This performs a loop in which the convolutional layer is modified in X Y Z dimension until it finds a good fit. 
	#The first IF condition will be true only in the last interaction, its last instruction will take it out from the while. 
	while layer_done == 0:
 		#minimum words to transfer in order to compute a full row (line) of the output
		minimum_2fill_output_row = layer[0]*kernel*layer[2] 
		#if the miminum is smaller than the storage of the subarray, no partial convolutions is required, 
		#and the input will be tiled in 2D only. 
		if minimum_2fill_output_row < words16b_sub:

			#this block calculates how many rows we can fit in one subarray, and the amount of output linked to i
			words_per_input_row = layer[0]*layer[2]
			rows = math.trunc(words16b_sub/words_per_input_row)
			if rows > _layer_.input[1]:
				rows =  _layer_.input[1]
			input_depth_reduction_factor = int(layer_original_depth/layer[2])
			output_rows = rows - kernel + 1
			total_words = rows*words_per_input_row

			input_block = [layer[0],rows,layer[2]]
			output_block = [output_layer[0], output_rows, _layer_.output_filter]  

			output_words_per_block = output_layer[0]*output_rows*_layer_.output_filter
			## END of FUNCTION, if this pointed is reached, the tilling is done. 
			layer_done = 1

		#If I cannot fit the miminum amount of rows to calculate one row of output, I will perform partial convolutions. 
		else:
			flag_layer_modif = 1

			#this block modifies the structure of convoltion, decreasing the depth of the input and kernels. 
			if output_layer[0] % output_dimensions_reduction_factor == 0 and output_dimensions_reduction_factor < layer[0]:
				output_layer[0] = int(output_layer_original_dimension/output_dimensions_reduction_factor)
				layer[0] = output_layer[0] + kernel - 1

			output_dimensions_reduction_factor += 1
			
		
			if output_dimensions_reduction_factor > layer[0] :			
				if layer_original_depth % input_depth_reduction_factor == 0:
						layer[2] = int(layer_original_depth/input_depth_reduction_factor)
						output_dimensions_reduction_factor = 1
						layer[0] = input_layer_original_dimension
				
				input_depth_reduction_factor += 1


	return[input_block, output_block, total_words]



#This function modulates the size of words in the subarray, decreasing one word per loop in while. 
#The ideia is that artifically I can create perfect fit for multiple tiles. 
def layer_tilling_coherence(_layer_, words16b_sub):

	output_layer_tilling = layer_tilling(_layer_, words16b_sub)

	total_input_words = _layer_.input[0]*_layer_.input[1]*_layer_.input[2]
	output_layer = _layer_.get_output()
	total_output_words = output_layer[0]*output_layer[1]*output_layer[2]


	rounds = 0 
	while rounds == 0:
		words16b_sub -= 1 
		output_layer_tilling = layer_tilling(_layer_, words16b_sub)
		input_depth_reduction_factor = int(_layer_.input[2]/output_layer_tilling[0][2])
		stream_in = output_layer_tilling[0][0]*output_layer_tilling[0][1]*output_layer_tilling[0][2]
		stream_out = output_layer_tilling[1][0]*output_layer_tilling[1][1]*output_layer_tilling[1][2]
	
		if (total_output_words*input_depth_reduction_factor % stream_out == 0) and (_layer_.get_output()[1]%output_layer_tilling[1][1] == 0): 
			rounds =  int(total_output_words*input_depth_reduction_factor/stream_out)	
		else: 
			rounds = 0

		output_layer_tilling.append(rounds)
	return(output_layer_tilling)




#top level function for the non-naive mapping. This function alters the structure of the tiles to be square shaped, reducing data transfers
def layer_squaring(_layer_, words16b_sub):
	kernel = _layer_.kernel
	outputd = _layer_.get_output()[0]
	output_layer_tilling = layer_tilling_coherence(_layer_,words16b_sub)

	X = output_layer_tilling[1][0]
	x_final = X
	Y = output_layer_tilling[1][1]
	y_final = Y
	input_words_2bcomp_smaller = output_layer_tilling[0][0]*output_layer_tilling[0][1]*output_layer_tilling[0][2]
	res = X*Y
	for yi in range(X+1):
		for xi in range(X+1):
			if yi*xi == res:
				input_words_2bcomp = (yi+kernel-1)*(xi+kernel-1)*output_layer_tilling[0][2]

				if(input_words_2bcomp < input_words_2bcomp_smaller) and (outputd % xi == 0) and (outputd % yi == 0) :
					input_words_2bcomp_smaller = input_words_2bcomp
					x_final = xi
					y_final  = yi
	output_layer_tilling[1][0] = x_final
	output_layer_tilling[1][1] = y_final
	output_layer_tilling[2] = input_words_2bcomp_smaller
	output_layer_tilling[0][0] = x_final + kernel - 1
	output_layer_tilling[0][1] = y_final + kernel - 1

	return output_layer_tilling


#top level function for the naive mapping. The idea with this mapping 
#is that each subarray will receive the miminum ammount of words required
#to calculate a single output. This maximizes parallelization at the cost 
#of incresed data transfers. 
def naive_layer_squaring(_layer_, words16b_sub):

	output = _layer_.get_output()
	min_block = _layer_.kernel * _layer_.kernel * _layer_.input[2]
	inputf = _layer_.input[2]
	divider = 1
	
	input_block = [_layer_.kernel,_layer_.kernel,inputf]
	output_block = [1,1,_layer_.output_filter]
	
	while min_block > words16b_sub:
		divider += 1
		if _layer_.input[2] % divider == 0:
			inputf = int(_layer_.input[2]/divider)
			min_block = _layer_.kernel * _layer_.kernel * inputf

			input_block = [_layer_.kernel,_layer_.kernel,inputf]
			output_block = [1,1,_layer_.output_filter]


	rounds = output[0]*output[1]*divider
	total_words = int(_layer_.kernel * _layer_.kernel * inputf)
	
	return([input_block, output_block, total_words, rounds])

	

def top_level_layer_tilling(_layer_, print_en, words16b_sub, naive_or_not):
	if naive_or_not == 1:
		s = naive_layer_squaring(_layer_,words16b_sub)
	else:
		s = layer_squaring(_layer_,words16b_sub)
		


	In_Z_ratio = (_layer_.input[2]/s[0][2])
	Out_Y_ratio = (_layer_.get_output()[1]/s[1][1])
	Out_X_ratio = (_layer_.get_output()[0]/s[1][0])
	kz = s[0][2]

	#ops[0]--> Multiplications
	#ops[1]--> stream in
	#ops[2]--> stream out
	ops = [_layer_.kernel*_layer_.kernel*kz*s[1][0]*s[1][1]*s[1][2], s[0][0]*s[0][1]*s[0][2], s[1][0]*s[1][1]*s[1][2]]

	

	if(print_en): print("\n--------------------------------------------------------------------")
	if (_layer_.type == "Conv") and (_layer_.kernel != 1):
		if(print_en): print("----------------------  Regular Convolution ------------------------")
	elif (_layer_.type == "SeparableConv"):
		if(print_en): print("----------------------  Depthwise Convolution ----------------------")
		s[1][2] = s[0][2]
		output_words_per_block = s[1][0]*s[1][1]*s[1][2]
		ops = [_layer_.kernel*_layer_.kernel*output_words_per_block, s[0][0]*s[0][1]*s[0][2], s[1][0]*s[1][1]*s[1][2]]
	elif (_layer_.kernel == 1):
		if(print_en): print("----------------------  Pointwise Convolution ----------------------")
	if(print_en): print("--------------------------------------------------------------------")


	if print_en == 1:

		check_ratios(Out_X_ratio, Out_Y_ratio, In_Z_ratio)

		print("Input layer:", _layer_.input, "Output layer:", _layer_.get_output())
		print("Input tile:", s[0], "Output tile:", s[1] , "Kernel:", (_layer_.kernel,_layer_.kernel,kz))

		print("\nOutput X ratio =", Out_X_ratio, "\nOutput Y ratio =", Out_Y_ratio,"\nInput Z ratio =", In_Z_ratio)
		
		print("\nAmount of tiles:", s[3])
		print("Maximum number of subarrays in parallel: ", int(Out_Y_ratio*Out_X_ratio) )
		
		print("\nMultiply operations per block:", ops[0])
		print("Stream in per block:", ops[1])
		print("Stream out per block:", ops[2])

	return(s, ops)

