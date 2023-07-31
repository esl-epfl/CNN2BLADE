# --------------------------------------------------------------------------------
# -- Title       : convolutional layer class definition 
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : Conv_Class.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 9 2021
# --------------------------------------------------------------------------------
# -- Module description :  This file defines the class for 1 convolutional layer
# --------------------------------------------------------------------------------



class conv_layer: 
	def __init__(self, i=(0,0,0), of=1, k=3,s=1, t="Conv", wq = 8, aq = 16):
		self.input = i
		self.output_filter = of
		self.kernel = k
		self.stride = s
		self.type = t
		self.weightQ = wq
		self.activQ = aq

    #return the dimensions of the output layer
	def get_output(self):
		output_d = int((self.input[0] - self.kernel + 1)/self.stride)
		return (output_d, output_d, self.output_filter)

	#return input data transfers
	def get_dtin(self):
		return (self.input[0] * self.input[1] * self.input[2])
	
	#return input data transfers
	def get_dtout(self):
		output_d = self.get_output()
		return (output_d[0] * output_d[1] * output_d[2])


