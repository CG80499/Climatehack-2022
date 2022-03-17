# My 0.748 Climatehack model 

The way my model works is it first applies a CNN to the 12 input images and concatenates the filters. It then passes the concatenated filters through a second CNN which upscales it into 24 64x64 images. The motivation behind this was that I had some experience creating neural networks that colour black and white images which work in a similar fashion.

I strongly advise using a pre-trained CNN for feature extraction (I used ResNet) this will provide better results with less training time. I also found that running batch normalisation during inferencing and applying Gaussian Blurring to the output image both improve performance.

If you wish to take this model further I would suggest using optical flow as an additional input source, trying different pre-trained CNNs for feature extraction, and reducing the size of the last kernel to 1. Another thing you could do is predict the difference between images rather than images themselves. But I found that this improves realism but not the score.
# Data Loading 

I found loading the data rather cubersome. So created a system the saves the numpy arrays in pickle files and loads them in "chunks" of about 1-4GB. I have included a link to about 30GB worth of these chunks. I have also inlcude a script that loads these chunks as needed in a PyTorch dataset object to avoid running out of RAM.
# Optical flow model based on Deepmind's Perceiver
<img width="661" alt="Screenshot 2022-03-16 175511" src="https://user-images.githubusercontent.com/94075036/158827253-42c30173-7d84-44d6-9dae-f7e3da8278a1.png">
*The first row is the predictions and the second row is the actual images.*

This model works similarly to optical flow in the included examples apart from that that instead of using an OpenCV implementation it uses Deepmind's Perceiver network. The results from using just optical flow are actually fairly good. I tried averaging these images with my CNN images and achieved fairly good results in testing granted I couldn't replicate them in DOXA. I also included a scipt that just computes the optical flow between 2 128x128 grayscale images. I think that you could get some very strong results by putting the optical flow from the perceiver into another network.