# --------------------------------------------------------------------------------
# -- Title       : Analysis of fully connected layers - batch analysis
# -- Project     : CNN2BLADE  
# --------------------------------------------------------------------------------
# -- File               : RunFClayers.py
# -- Author             : Marco Antonio Rios marco.rios@epfl.ch
# -- Company            : EPFL - ESL 
# -- Created     	    : Mon Aug 9 2021
# --------------------------------------------------------------------------------
# -- Module description : This module executes the tilling of all the FC layers. 
# --------------------------------------------------------------------------------


import sys, math, csv

MAC_cycles = 2
# --------------------------------------------------------------------------------
# Cycles MAC: cycles on the accumulations  
#           1 - TETC architecture 
#           2 - CODES architecture (MACL, MACH) 
# --------------------------------------------------------------------------------
WLP = True         #word-level parallelism enable
acceleration = 2.0 #average acceleration for the embedded shifts
cycles_op = 2 
    #2 cycles per operation in mult --> TETC
    #1 cycle per operation in mult --> CODES 

f = open('Results/' + "FClayers" + str(MAC_cycles) + "MAC.csv", 'w', newline='')
writer = csv.writer(f)

#process all the FC layers from the model list 
for model in range(1,7):
    #CNN - import parameters
    valid_model = 1 
    while valid_model == 1:
        valid_model = 0
        if model == 1:
            from models.googlenet import CNNname, fc_layer_list
        elif model == 2:
            from models.mobilenet import CNNname, fc_layer_list
        elif model == 3:
            from models.resnext   import CNNname, fc_layer_list
        elif model == 4:
            from models.alexnet   import CNNname, fc_layer_list
        elif model== 5:
            from models.vgg       import CNNname, fc_layer_list
        elif model== 6:
            from models.resnet8   import CNNname, fc_layer_list
        elif model== 7:
            from models.custom    import CNNname, fc_layer_list
        else:
            input("enter a valid CNN")
            valid_model = 1
    #print("CNN name:", CNNname)



    number_of_layers = len(fc_layer_list)
    data = []
    for layer in range(number_of_layers):
        stream_in = fc_layer_list[layer][0] * fc_layer_list[layer][1]             # input data transfer
        stream_out = fc_layer_list[layer][1]                                      # output data transfer      
        mults_per_output = fc_layer_list[layer][0]                                # amount of multiplication required by each output
        weightQ =  fc_layer_list[layer][2]                                        # weights (IMO) bitwidth
        activQ =  fc_layer_list[layer][3]                                         # activations (BO) bitwidth
        data.append([stream_in, stream_out, mults_per_output, weightQ, activQ])   # vector with all the data above for each FC layer



    #creates a vector with the considered number of subarrays, in this case, from 1 to 128.
    num_subs = []
    potencies = list(range(1,8))
    for pw in potencies:
        num_subs.append(2**pw)

    #for each FC layer, it maps in 1 up to 128 subarray
    for num in num_subs:        
        cycles_IMC = 0
        cycles_DT  = 0 
        for layer in range(number_of_layers):
            cycles_per_multiplication = math.ceil((data[layer][4] * cycles_op)/acceleration + MAC_cycles)
            # number of output divided by the number of subarrays. Rounding up.
            # for example, a FC layers with 100 outputs and being executed in 64 subarrays will require two rounds of computation. 
            rounds = math.ceil(data[layer][1]/num)
            mults_perout = data[layer][2]

            if data[layer][3] == 8 and WLP == True: 
                cycles_mults_full = mults_perout * rounds * cycles_per_multiplication / 2
                cycles_IMC += cycles_mults_full
                cycles_DT += data[layer][0]/2 + data[layer][1]*MAC_cycles/2
            else: 
                cycles_mults_full = mults_perout * rounds * cycles_per_multiplication 
                cycles_IMC += cycles_mults_full
                cycles_DT += data[layer][0] + data[layer][1]*MAC_cycles
        
        #line to written in the file
        #CNN name, number of subarrays, cycles IMC, cycles DT and total cycles
        row = [CNNname, str(num), str(int(cycles_IMC)),  str(int(cycles_DT)), str(int(cycles_IMC+cycles_DT))  ]
        writer.writerow(row)


f.close()