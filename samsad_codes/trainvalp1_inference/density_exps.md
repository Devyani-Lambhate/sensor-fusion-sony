## Baseline

```text
mAP: 0.6445
mATE: 0.2267
mASE: 0.2383
mAOE: 0.3074
mAVE: 0.3980
mAAE: 0.1075
NDS: 0.6945
Eval time: 3.8s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.906   0.156   0.149   0.054   0.340   0.236 
truck                   0.662   0.268   0.194   0.154   0.285   0.230 
bus                     0.743   0.365   0.138   0.121   0.681   0.135 
trailer                 0.221   0.215   0.131   0.073   0.663   0.000 
construction_vehicle    0.437   0.518   0.427   1.053   0.094   0.148 
pedestrian              0.908   0.125   0.292   0.417   0.206   0.063 
motorcycle              0.625   0.177   0.312   0.228   0.598   0.017 
bicycle                 0.657   0.163   0.216   0.616   0.316   0.031 
traffic_cone            0.875   0.102   0.281   nan     nan     nan   
barrier                 0.411   0.179   0.243   0.052   nan     nan 
```

## Beam Drop - 30

```text
mAP: 0.5958
mATE: 0.2305
mASE: 0.2397
mAOE: 0.3117
mAVE: 0.3860
mAAE: 0.1079
NDS: 0.6703
Eval time: 3.8s

Per-class results:
Object Class            AP      ATE     ASE     AOE     AVE     AAE   
car                     0.845   0.159   0.149   0.058   0.356   0.235 
truck                   0.605   0.279   0.199   0.143   0.294   0.218 
bus                     0.693   0.366   0.136   0.131   0.691   0.143 
trailer                 0.166   0.223   0.127   0.091   0.528   0.000 
construction_vehicle    0.406   0.522   0.433   1.066   0.098   0.147 
pedestrian              0.841   0.127   0.295   0.431   0.214   0.065 
motorcycle              0.563   0.181   0.306   0.229   0.591   0.024 
bicycle                 0.619   0.165   0.222   0.600   0.315   0.032 
traffic_cone            0.822   0.104   0.290   nan     nan     nan   
barrier                 0.397   0.179   0.240   0.055   nan     nan
```

## Beam Drop 60
```text
mAP: 0.5367
mATE: 0.2377
mASE: 0.2460
mAOE: 0.3330
mAVE: 0.3891
mAAE: 0.1106
NDS: 0.6367
Eval time: 4.0s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.769 	0.162 	0.149 	0.062 	0.377 	0.230 
truck               	0.547 	0.276 	0.202 	0.161 	0.280 	0.234 
bus                 	0.633 	0.384 	0.142 	0.138 	0.692 	0.131 
trailer             	0.179 	0.193 	0.135 	0.081 	0.469 	0.000 
construction_vehicle	0.335 	0.589 	0.465 	1.143 	0.107 	0.179 
pedestrian          	0.756 	0.127 	0.295 	0.430 	0.209 	0.066 
motorcycle          	0.507 	0.202 	0.312 	0.267 	0.669 	0.019 
bicycle             	0.536 	0.160 	0.235 	0.660 	0.309 	0.025 
traffic_cone        	0.746 	0.104 	0.291 	nan   	nan   	nan   
barrier             	0.360 	0.180 	0.235 	0.055 	nan   	nan 
```

## Beam Drop 90

```text
mAP: 0.4934
mATE: 0.2357
mASE: 0.2481
mAOE: 0.3122
mAVE: 0.3802
mAAE: 0.1132
NDS: 0.6177
Eval time: 4.1s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.679 	0.165 	0.151 	0.066 	0.373 	0.235 
truck               	0.486 	0.286 	0.209 	0.158 	0.310 	0.235 
bus                 	0.560 	0.366 	0.146 	0.132 	0.649 	0.159 
trailer             	0.238 	0.237 	0.149 	0.093 	0.560 	0.000 
construction_vehicle	0.328 	0.527 	0.448 	1.005 	0.087 	0.147 
pedestrian          	0.675 	0.132 	0.295 	0.455 	0.209 	0.066 
motorcycle          	0.485 	0.170 	0.310 	0.226 	0.557 	0.025 
bicycle             	0.481 	0.167 	0.239 	0.618 	0.297 	0.039 
traffic_cone        	0.671 	0.108 	0.297 	nan   	nan   	nan   
barrier             	0.330 	0.199 	0.237 	0.055 	nan   	nan
```

## Beam Drop 180
```text
mAP: 0.2767
mATE: 0.2593
mASE: 0.2652
mAOE: 0.3299
mAVE: 0.5344
mAAE: 0.1154
NDS: 0.4879
Eval time: 4.3s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.466 	0.183 	0.162 	0.083 	0.445 	0.221 
truck               	0.296 	0.297 	0.222 	0.165 	0.348 	0.231 
bus                 	0.341 	0.368 	0.165 	0.158 	0.747 	0.128 
trailer             	0.109 	0.336 	0.196 	0.069 	0.838 	0.000 
construction_vehicle	0.195 	0.547 	0.468 	0.990 	0.120 	0.179 
pedestrian          	0.409 	0.138 	0.302 	0.430 	0.226 	0.077 
motorcycle          	0.051 	0.217 	0.307 	0.353 	1.211 	0.053 
bicycle             	0.311 	0.186 	0.265 	0.658 	0.340 	0.035 
traffic_cone        	0.391 	0.117 	0.315 	nan   	nan   	nan   
barrier             	0.197 	0.204 	0.251 	0.064 	nan   	nan
```

## Channel Drop 8
```text
mAP: 0.5803
mATE: 0.2463
mASE: 0.2483
mAOE: 0.3297
mAVE: 0.4649
mAAE: 0.1037
NDS: 0.6509
Eval time: 4.0s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.825 	0.169 	0.155 	0.072 	0.365 	0.237 
truck               	0.585 	0.282 	0.205 	0.170 	0.289 	0.209 
bus                 	0.713 	0.361 	0.143 	0.094 	0.777 	0.123 
trailer             	0.194 	0.283 	0.173 	0.199 	0.961 	0.000 
construction_vehicle	0.377 	0.567 	0.457 	1.207 	0.096 	0.152 
pedestrian          	0.837 	0.129 	0.290 	0.429 	0.213 	0.069 
motorcycle          	0.500 	0.202 	0.307 	0.161 	0.718 	0.023 
bicycle             	0.562 	0.165 	0.221 	0.578 	0.300 	0.017 
traffic_cone        	0.824 	0.108 	0.292 	nan   	nan   	nan   
barrier             	0.387 	0.198 	0.239 	0.058 	nan   	nan   

```


## Channel Drop 16

```text
mAP: 0.4750
mATE: 0.2748
mASE: 0.2712
mAOE: 0.3558
mAVE: 0.5094
mAAE: 0.1119
NDS: 0.5852
Eval time: 4.0s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.680 	0.194 	0.166 	0.123 	0.405 	0.238 
truck               	0.434 	0.321 	0.239 	0.220 	0.274 	0.203 
bus                 	0.602 	0.403 	0.143 	0.091 	0.807 	0.143 
trailer             	0.214 	0.303 	0.219 	0.148 	1.001 	0.000 
construction_vehicle	0.274 	0.615 	0.517 	1.207 	0.110 	0.150 
pedestrian          	0.685 	0.149 	0.300 	0.439 	0.232 	0.076 
motorcycle          	0.323 	0.232 	0.322 	0.270 	0.852 	0.044 
bicycle             	0.393 	0.188 	0.248 	0.651 	0.394 	0.041 
traffic_cone        	0.730 	0.120 	0.327 	nan   	nan   	nan   
barrier             	0.414 	0.222 	0.233 	0.055 	nan   	nan
```

## Channel Drop 20

```text
mAP: 0.3775
mATE: 0.3038
mASE: 0.2913
mAOE: 0.3820
mAVE: 0.5548
mAAE: 0.1194
NDS: 0.5236
Eval time: 4.2s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.575 	0.211 	0.177 	0.153 	0.448 	0.234 
truck               	0.354 	0.375 	0.266 	0.215 	0.294 	0.210 
bus                 	0.467 	0.449 	0.158 	0.086 	0.981 	0.125 
trailer             	0.147 	0.311 	0.264 	0.301 	1.189 	0.002 
construction_vehicle	0.187 	0.629 	0.542 	1.145 	0.117 	0.171 
pedestrian          	0.585 	0.164 	0.305 	0.447 	0.248 	0.079 
motorcycle          	0.146 	0.301 	0.350 	0.318 	0.752 	0.095 
bicycle             	0.292 	0.197 	0.263 	0.710 	0.411 	0.038 
traffic_cone        	0.657 	0.126 	0.354 	nan   	nan   	nan   
barrier             	0.363 	0.276 	0.235 	0.062 	nan   	nan
```

## Channel Drop 24

```text
mAP: 0.2468
mATE: 0.3692
mASE: 0.3171
mAOE: 0.5669
mAVE: 0.7374
mAAE: 0.1373
NDS: 0.4106
Eval time: 4.6s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.434 	0.252 	0.197 	0.248 	0.536 	0.230 
truck               	0.223 	0.433 	0.310 	0.302 	0.365 	0.203 
bus                 	0.325 	0.513 	0.175 	0.227 	1.135 	0.150 
trailer             	0.004 	0.502 	0.304 	1.299 	1.738 	0.000 
construction_vehicle	0.106 	0.816 	0.626 	1.380 	0.134 	0.213 
pedestrian          	0.395 	0.207 	0.323 	0.491 	0.284 	0.106 
motorcycle          	0.007 	0.322 	0.331 	0.444 	1.275 	0.174 
bicycle             	0.149 	0.211 	0.294 	0.654 	0.433 	0.022 
traffic_cone        	0.514 	0.155 	0.385 	nan   	nan   	nan   
barrier             	0.311 	0.280 	0.227 	0.058 	nan   	nan
```

## Loss of Echo 0.25

```text
mAP: 0.5564
mATE: 0.2402
mASE: 0.2401
mAOE: 0.3363
mAVE: 0.4515
mAAE: 0.1114
NDS: 0.6403
Eval time: 3.9s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.764 	0.173 	0.153 	0.072 	0.382 	0.222 
truck               	0.511 	0.276 	0.198 	0.156 	0.304 	0.234 
bus                 	0.633 	0.380 	0.141 	0.152 	0.707 	0.152 
trailer             	0.187 	0.227 	0.120 	0.082 	0.861 	0.002 
construction_vehicle	0.296 	0.554 	0.436 	1.176 	0.098 	0.170 
pedestrian          	0.909 	0.125 	0.292 	0.415 	0.206 	0.063 
motorcycle          	0.452 	0.216 	0.314 	0.257 	0.730 	0.021 
bicycle             	0.533 	0.171 	0.218 	0.663 	0.324 	0.026 
traffic_cone        	0.871 	0.102 	0.286 	nan   	nan   	nan   
barrier             	0.408 	0.179 	0.243 	0.053 	nan   	nan   
```

## Loss of Echo 0.50

```text
mAP: 0.4499
mATE: 0.2487
mASE: 0.2477
mAOE: 0.3435
mAVE: 0.4353
mAAE: 0.1158
NDS: 0.5858
Eval time: 3.9s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.609 	0.206 	0.158 	0.106 	0.458 	0.213 
truck               	0.338 	0.290 	0.197 	0.163 	0.343 	0.221 
bus                 	0.484 	0.382 	0.143 	0.219 	0.887 	0.156 
trailer             	0.083 	0.154 	0.139 	0.135 	0.296 	0.000 
construction_vehicle	0.158 	0.555 	0.435 	1.041 	0.116 	0.211 
pedestrian          	0.909 	0.125 	0.292 	0.416 	0.205 	0.064 
motorcycle          	0.278 	0.278 	0.340 	0.237 	0.884 	0.029 
bicycle             	0.366 	0.214 	0.231 	0.725 	0.294 	0.033 
traffic_cone        	0.868 	0.103 	0.299 	nan   	nan   	nan   
barrier             	0.407 	0.180 	0.244 	0.049 	nan   	nan
```

## Loss of Echo 0.75

```text
mAP: 0.3521
mATE: 0.3218
mASE: 0.2601
mAOE: 0.3875
mAVE: 0.7087
mAAE: 0.1170
NDS: 0.4965
Eval time: 3.6s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.417 	0.285 	0.168 	0.190 	0.603 	0.180 
truck               	0.154 	0.353 	0.219 	0.252 	0.419 	0.180 
bus                 	0.352 	0.414 	0.144 	0.097 	0.940 	0.080 
trailer             	0.013 	0.446 	0.165 	0.372 	2.009 	0.185 
construction_vehicle	0.064 	0.627 	0.487 	1.143 	0.106 	0.158 
pedestrian          	0.911 	0.125 	0.292 	0.411 	0.205 	0.063 
motorcycle          	0.163 	0.394 	0.344 	0.214 	0.996 	0.063 
bicycle             	0.175 	0.296 	0.237 	0.756 	0.392 	0.026 
traffic_cone        	0.864 	0.103 	0.301 	nan   	nan   	nan   
barrier             	0.406 	0.175 	0.244 	0.053 	nan   	nan
```

## Loss of Echo 0.90

```text
mAP: 0.3064
mATE: 0.3498
mASE: 0.2599
mAOE: 0.3939
mAVE: 0.9285
mAAE: 0.1269
NDS: 0.4473
Eval time: 3.6s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.289 	0.386 	0.180 	0.319 	0.757 	0.151 
truck               	0.041 	0.514 	0.249 	0.424 	0.728 	0.209 
bus                 	0.321 	0.419 	0.143 	0.132 	1.057 	0.079 
trailer             	0.008 	0.351 	0.115 	0.319 	3.096 	0.284 
construction_vehicle	0.009 	0.665 	0.488 	0.865 	0.079 	0.171 
pedestrian          	0.912 	0.125 	0.293 	0.410 	0.205 	0.063 
motorcycle          	0.105 	0.399 	0.331 	0.156 	1.047 	0.027 
bicycle             	0.114 	0.359 	0.245 	0.868 	0.460 	0.031 
traffic_cone        	0.861 	0.103 	0.309 	nan   	nan   	nan   
barrier             	0.405 	0.176 	0.246 	0.052 	nan   	nan
```

## Object Noise 0.02

```text
mAP: 0.6446
mATE: 0.2262
mASE: 0.2376
mAOE: 0.3075
mAVE: 0.3961
mAAE: 0.1079
NDS: 0.6948
Eval time: 3.7s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.906 	0.155 	0.149 	0.055 	0.338 	0.236 
truck               	0.665 	0.264 	0.193 	0.151 	0.291 	0.232 
bus                 	0.739 	0.363 	0.138 	0.143 	0.679 	0.133 
trailer             	0.223 	0.222 	0.129 	0.076 	0.674 	0.000 
construction_vehicle	0.432 	0.513 	0.421 	1.049 	0.095 	0.146 
pedestrian          	0.907 	0.125 	0.292 	0.417 	0.206 	0.063 
motorcycle          	0.627 	0.177 	0.312 	0.224 	0.572 	0.023 
bicycle             	0.660 	0.161 	0.216 	0.600 	0.313 	0.031 
traffic_cone        	0.875 	0.102 	0.282 	nan   	nan   	nan   
barrier             	0.412 	0.178 	0.243 	0.052 	nan   	nan
```

## Object Noise 0.05

```text
mAP: 0.6456
mATE: 0.2286
mASE: 0.2389
mAOE: 0.3050
mAVE: 0.4042
mAAE: 0.1078
NDS: 0.6943
Eval time: 3.9s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.907 	0.157 	0.150 	0.059 	0.338 	0.231 
truck               	0.666 	0.256 	0.191 	0.155 	0.286 	0.234 
bus                 	0.739 	0.359 	0.139 	0.146 	0.719 	0.136 
trailer             	0.220 	0.249 	0.137 	0.078 	0.711 	0.000 
construction_vehicle	0.439 	0.522 	0.421 	1.058 	0.094 	0.150 
pedestrian          	0.907 	0.125 	0.292 	0.417 	0.206 	0.063 
motorcycle          	0.630 	0.176 	0.313 	0.143 	0.571 	0.020 
bicycle             	0.662 	0.161 	0.220 	0.636 	0.309 	0.028 
traffic_cone        	0.874 	0.102 	0.282 	nan   	nan   	nan   
barrier             	0.411 	0.178 	0.243 	0.053 	nan   	nan
```

## Object Noise 0.2

```text
mAP: 0.6309
mATE: 0.2418
mASE: 0.2453
mAOE: 0.3099
mAVE: 0.4595
mAAE: 0.1171
NDS: 0.6781
Eval time: 3.7s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.891 	0.183 	0.155 	0.080 	0.394 	0.221 
truck               	0.612 	0.272 	0.191 	0.163 	0.392 	0.284 
bus                 	0.735 	0.370 	0.140 	0.059 	0.768 	0.137 
trailer             	0.237 	0.261 	0.161 	0.092 	0.827 	0.000 
construction_vehicle	0.399 	0.563 	0.436 	1.080 	0.101 	0.190 
pedestrian          	0.907 	0.125 	0.292 	0.417 	0.206 	0.062 
motorcycle          	0.593 	0.196 	0.321 	0.159 	0.708 	0.021 
bicycle             	0.656 	0.170 	0.228 	0.680 	0.281 	0.022 
traffic_cone        	0.872 	0.101 	0.285 	nan   	nan   	nan   
barrier             	0.407 	0.177 	0.244 	0.058 	nan   	nan
```

## Object Noise 0.5

```text
mAP: 0.5730
mATE: 0.2995
mASE: 0.2573
mAOE: 0.3989
mAVE: 0.6540
mAAE: 0.1383
NDS: 0.6117
Eval time: 3.8s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.824 	0.256 	0.164 	0.101 	0.812 	0.206 
truck               	0.502 	0.358 	0.201 	0.246 	1.088 	0.377 
bus                 	0.611 	0.429 	0.153 	0.177 	1.289 	0.119 
trailer             	0.230 	0.401 	0.187 	0.336 	0.531 	0.000 
construction_vehicle	0.304 	0.677 	0.484 	1.101 	0.122 	0.260 
pedestrian          	0.906 	0.125 	0.291 	0.420 	0.205 	0.063 
motorcycle          	0.450 	0.259 	0.323 	0.371 	0.859 	0.052 
bicycle             	0.636 	0.206 	0.231 	0.782 	0.326 	0.030 
traffic_cone        	0.869 	0.103 	0.293 	nan   	nan   	nan   
barrier             	0.397 	0.181 	0.245 	0.055 	nan   	nan
```

## Object Noise 1.0

```text
mAP: 0.4826
mATE: 0.3642
mASE: 0.2725
mAOE: 0.5002
mAVE: 0.7807
mAAE: 0.1455
NDS: 0.5350
Eval time: 3.8s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.712 	0.316 	0.171 	0.173 	1.051 	0.188 
truck               	0.302 	0.462 	0.253 	0.400 	1.099 	0.309 
bus                 	0.378 	0.657 	0.171 	0.208 	1.729 	0.195 
trailer             	0.188 	0.400 	0.196 	0.714 	0.388 	0.000 
construction_vehicle	0.203 	0.840 	0.548 	1.252 	0.233 	0.341 
pedestrian          	0.903 	0.126 	0.292 	0.421 	0.205 	0.062 
motorcycle          	0.317 	0.297 	0.320 	0.491 	1.059 	0.041 
bicycle             	0.570 	0.258 	0.230 	0.785 	0.481 	0.028 
traffic_cone        	0.862 	0.103 	0.300 	nan   	nan   	nan   
barrier             	0.391 	0.183 	0.245 	0.056 	nan   	nan 
```

## Object Noise 2.0

```text
mAP: 0.4351
mATE: 0.3713
mASE: 0.2793
mAOE: 0.4911
mAVE: 0.7135
mAAE: 0.1331
NDS: 0.5187
Eval time: 3.8s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.593 	0.316 	0.176 	0.179 	0.993 	0.161 
truck               	0.178 	0.507 	0.266 	0.470 	0.819 	0.281 
bus                 	0.274 	0.630 	0.182 	0.337 	1.922 	0.211 
trailer             	0.124 	0.442 	0.212 	0.704 	0.246 	0.000 
construction_vehicle	0.220 	0.848 	0.548 	1.149 	0.200 	0.259 
pedestrian          	0.899 	0.126 	0.294 	0.416 	0.205 	0.061 
motorcycle          	0.308 	0.298 	0.324 	0.403 	0.855 	0.062 
bicycle             	0.529 	0.256 	0.236 	0.711 	0.468 	0.030 
traffic_cone        	0.854 	0.104 	0.303 	nan   	nan   	nan   
barrier             	0.372 	0.185 	0.252 	0.052 	nan   	nan
```

## Scene Noise 0.1

```text
mAP: 0.6034
mATE: 0.2504
mASE: 0.2538
mAOE: 0.3150
mAVE: 0.5131
mAAE: 0.1102
NDS: 0.6574
Eval time: 3.8s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.890 	0.171 	0.174 	0.085 	0.386 	0.229 
truck               	0.627 	0.295 	0.189 	0.135 	0.371 	0.240 
bus                 	0.745 	0.364 	0.158 	0.050 	0.860 	0.121 
trailer             	0.131 	0.279 	0.166 	0.129 	0.856 	0.000 
construction_vehicle	0.357 	0.531 	0.418 	1.133 	0.109 	0.184 
pedestrian          	0.870 	0.134 	0.313 	0.387 	0.256 	0.065 
motorcycle          	0.484 	0.186 	0.329 	0.120 	0.889 	0.019 
bicycle             	0.595 	0.174 	0.206 	0.737 	0.377 	0.025 
traffic_cone        	0.803 	0.128 	0.312 	nan   	nan   	nan   
barrier             	0.530 	0.239 	0.272 	0.059 	nan   	nan
```

## Scene Noise 0.2

```text
mAP: 0.4195
mATE: 0.3285
mASE: 0.2782
mAOE: 0.4125
mAVE: 0.8103
mAAE: 0.1562
NDS: 0.5112
Eval time: 4.3s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.833 	0.217 	0.202 	0.102 	0.681 	0.254 
truck               	0.499 	0.338 	0.189 	0.234 	0.867 	0.397 
bus                 	0.672 	0.402 	0.183 	0.036 	1.097 	0.150 
trailer             	0.023 	0.437 	0.202 	0.765 	1.148 	0.042 
construction_vehicle	0.268 	0.593 	0.429 	1.120 	0.100 	0.188 
pedestrian          	0.598 	0.211 	0.358 	0.338 	0.415 	0.070 
motorcycle          	0.178 	0.246 	0.325 	0.374 	1.680 	0.128 
bicycle             	0.361 	0.218 	0.280 	0.657 	0.496 	0.020 
traffic_cone        	0.556 	0.253 	0.329 	nan   	nan   	nan   
barrier             	0.206 	0.372 	0.286 	0.087 	nan   	nan
```

## Scene Noise 0.5

```text
mAP: 0.1413
mATE: 0.6374
mASE: 0.3007
mAOE: 0.6191
mAVE: 1.9642
mAAE: 0.2741
NDS: 0.2875
Eval time: 4.6s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.479 	0.534 	0.196 	0.344 	3.028 	0.338 
truck               	0.191 	0.587 	0.312 	1.015 	3.212 	0.622 
bus                 	0.306 	0.715 	0.196 	0.270 	2.658 	0.136 
trailer             	0.020 	0.747 	0.231 	0.314 	1.075 	0.051 
construction_vehicle	0.011 	1.109 	0.580 	1.243 	0.734 	0.563 
pedestrian          	0.108 	0.577 	0.313 	0.963 	0.906 	0.179 
motorcycle          	0.003 	0.403 	0.322 	0.504 	3.266 	0.243 
bicycle             	0.040 	0.389 	0.232 	0.742 	0.836 	0.061 
traffic_cone        	0.220 	0.590 	0.362 	nan   	nan   	nan   
barrier             	0.036 	0.723 	0.264 	0.175 	nan   	nan
```

## Scene Noise 1.0

```text
mAP: 0.0394
mATE: 0.9466
mASE: 0.3384
mAOE: 0.9232
mAVE: 3.5631
mAAE: 0.3229
NDS: 0.1666
Eval time: 4.5s

Per-class results:
Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
car                 	0.199 	0.909 	0.203 	0.529 	5.088 	0.375 
truck               	0.028 	0.972 	0.391 	1.202 	4.880 	0.637 
bus                 	0.073 	1.100 	0.239 	1.327 	5.938 	0.230 
trailer             	0.000 	0.909 	0.368 	0.142 	3.270 	0.289 
construction_vehicle	0.000 	1.153 	0.556 	1.476 	1.751 	0.551 
pedestrian          	0.015 	0.951 	0.322 	1.303 	1.175 	0.327 
motorcycle          	0.002 	0.674 	0.288 	0.824 	4.879 	0.063 
bicycle             	0.004 	0.859 	0.268 	1.154 	1.525 	0.112 
traffic_cone        	0.068 	0.919 	0.410 	nan   	nan   	nan   
barrier             	0.005 	1.019 	0.338 	0.351 	nan   	nan
```