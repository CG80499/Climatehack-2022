from optical_flow import get_optical_flow
import torch

img1 = torch.rand(128, 128)
img2 = torch.rand(128, 128)

img1 = img1*2-1 #NORMALISE
img2 = img2*2-1

flow = get_optical_flow(img1, img2) #Returns flow in form (2, 128, 128)


