import torch, torchvision
import cv2
import numpy as np
from optical_flow import get_optical_flow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on: {}".format(str(device).upper()))

def remap_image(image, flow):
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)
    remap[..., 1] += np.arange(height)[:, np.newaxis]
    remapped_image = cv2.remap(src=image, map1=remap, map2=None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return remapped_image

def weighted_average(flows):
    w = [i**2 for i in range(1, 11+1)] #Try changing the weighting.
    return np.average(flows, axis=0, weights=w).astype(np.float32)

def get_flow_images(input_images):
    input_images = input_images*2-1
    input_images = torch.from_numpy(input_images)
    input_images = input_images.to(device)

    flows = []
    for i in range(11):
        flow = get_optical_flow(input_images[i], input_images[i+1], device).cpu().detach().numpy()
        flows.append(flow)
    flows = np.stack(flows)


    flow = weighted_average(flows)
    flow = flow.reshape(128, 128, 2)

    for i in range(24):
        image = input_images[-1].cpu().detach().numpy()
        image = (image+1)/2

        output_image = remap_image(image, flow*i).reshape(1, 128, 128)*2-1
        output_image = torch.from_numpy(output_image).to(device)
        input_images = torch.cat((input_images, output_image))

    

    input_images = (input_images+1)/2 #Unnormalise
    input_images = input_images.view([36, 1, 128, 128])
    input_images = torchvision.transforms.CenterCrop(64)(input_images)
    input_images[14:] = torchvision.transforms.GaussianBlur(kernel_size=(63, 63), sigma=3)(input_images[14:]) #Reduces jagged edges
    input_images = input_images.view([36, 64, 64])
 
    prediction = input_images[12:]
    return prediction







