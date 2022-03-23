import torch
from core.model import FairMOT as build_model

input_random = torch.zeros((1, 3,512, 512))

model = build_model(3, 1000)
output = model(input_random)

for i in range(len(output)):
    print(output[i].shape)
