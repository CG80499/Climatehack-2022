from transformers import PerceiverForOpticalFlow
import torch, torchvision
import math
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

#IMPORTANT: You need to save the model locally before submitting.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# source: https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/9
def extract_image_patches(x, kernel, device, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    x.to(device)
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches.to(device)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    patches.to(device)
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

def preprocess_image(image):
    image = torchvision.transforms.Resize((368, 496))(image)
    image = torch.cat((image, image, image), axis=0)
    return image.view([1, 3, 368, 496])

def get_optical_flow(img1, img2):
    """
    Takes as input Tensors of the form (128, 128).
    Return Optical flow of the form (2, 128, 128)
    The images should be normalised in the range [-1, 1]
    """
    img1 = preprocess_image(img1.view([1, 128, 128]))
    img2 = preprocess_image(img2.view([1, 128, 128]))

    img1.to(device)
    img2.to(device)

    input_images = torch.cat((img1, img2))
    input_images.to(device)
    patches = extract_image_patches(input_images, kernel=3, device=device)
    patches = patches.view(1, 2, 27, 368, 496)

    patches.to(device)

    with torch.no_grad():
        output = model(inputs=patches).logits

    output = output.view(1, 2, 368, 496)
    output = torchvision.transforms.Resize((128, 128))(output)
    return output.to(device)




