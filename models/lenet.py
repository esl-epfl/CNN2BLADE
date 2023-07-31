CNNname = "lenet"

conv_layer_list = [
[(32,32,1),   6, 5, 1, "Conv", 6,8],
[(14,14,6),  16, 5, 1, "Conv", 7,8],
[(5,5,16),  120, 5, 1, "Conv", 6,8],
]


fc_layer_list = [
[120, 84, 8, 6],
[84,  10, 8, 8],
]