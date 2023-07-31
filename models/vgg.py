CNNname = "vgg"


conv_layer_list = [
[(34,34,3),   64, 3, 1, "Conv", 8, 8],
[(34,34,64),  64, 3, 1, "Conv", 7, 8],
[(18,18,64), 128, 3, 1, "Conv", 8, 8],
[(18,18,128),128, 3, 1, "Conv", 7, 8],
[(10,10,128),256, 3, 1, "Conv", 8, 8],
[(10,10,256),256, 3, 1, "Conv", 8, 8],
[(10,10,256),256, 3, 1, "Conv", 8, 8],
[(6,6,256),  512, 3, 1, "Conv", 8, 8],
[(6,6,512),  512, 3, 1, "Conv", 7, 8],
[(6,6,512),  512, 3, 1, "Conv", 8, 8],
[(4,4,512),  512, 3, 1, "Conv", 8, 8],
[(4,4,512),  512, 3, 1, "Conv", 8, 8],
[(4,4,512),  512, 3, 1, "Conv", 8, 8],
]



fc_layer_list = [
[512,1024,8,6],
[1024,100,8,8]
]