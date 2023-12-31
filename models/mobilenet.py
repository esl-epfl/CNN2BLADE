CNNname = "mobilenet"

conv_layer_list = [
[(	32	,	32	,	3	),	32	,	3	,	1	,  "Conv"         , 8, 8],
[(	30	,	30	,	32	),	32	,	3	,	1	,  "SeparableConv", 8, 8],
[(	28	,	28	,	32	),	64	,	1	,	1	,  "Conv"         , 8, 8],
[(	28	,	28	,	64	),	64	,	3	,	1	,  "SeparableConv", 8, 8],
[(	26	,	26	,	64	),	128	,	1	,	1	,  "Conv"         , 8, 8],
[(	26	,	26	,	128	),	128	,	3	,	1	,  "SeparableConv", 8, 8],
[(	24	,	24	,	128	),	128	,	1	,	1	,  "Conv"         , 8, 8],
[(	24	,	24	,	128	),	128	,	3	,	1	,  "SeparableConv", 8, 8],
[(	22	,	22	,	128	),	256	,	1	,	1	,  "Conv"         , 8, 8],
[(	22	,	22	,	256	),	256	,	3	,	1	,  "SeparableConv", 8, 8],
[(	20	,	20	,	256	),	256	,	1	,	1	,  "Conv"         , 6, 8],
[(	20	,	20	,	256	),	256	,	3	,	1	,  "SeparableConv", 8, 8],
[(	18	,	18	,	256	),	512	,	1	,	1	,  "Conv"         , 7, 8],
[(	18	,	18	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	16	,	16	,	512	),	512	,	1	,	1	,  "Conv"         , 6, 8],
[(	16	,	16	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	14	,	14	,	512	),	512	,	1	,	1	,  "Conv"         , 7, 8],
[(	14	,	14	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	12	,	12	,	512	),	512	,	1	,	1	,  "Conv"         , 7, 8],
[(	12	,	12	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	10	,	10	,	512	),	512	,	1	,	1	,  "Conv"         , 8, 8],
[(	10	,	10	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	8	,	8	,	512	),	512	,	1	,	1	,  "Conv"         , 8, 8],
[(	8	,	8	,	512	),	512	,	3	,	1	,  "SeparableConv", 8, 8],
[(	6	,	6	,	512	),	1024,	1	,	1	,  "Conv"         , 8, 8],
[(	6	,	6	,	1024),	1024,	3	,	1	,  "SeparableConv", 8, 8],
[(	4	,	4	,	1024),	1024,	1	,	1	,  "Conv"         , 8, 8],
]


fc_layer_list = [
[1024,100,8,7]
]

