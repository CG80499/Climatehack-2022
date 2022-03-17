import torch, torchvision
import pickle

resnet = torchvision.models.resnet34(pretrained=False)
#resnet.eval()

def preprocess_sample(sample):
    sample = torchvision.transforms.Resize(224)(sample)
    sample = torch.cat((sample, sample, sample), axis=1)
    sample = torchvision.transforms.Normalize((0.4302, 0.4575, 0.4539), (0.2361, 0.2347, 0.2432))(sample)
    return sample

with open("batch2_size8.pickle", 'rb') as f:
    batch_data = pickle.load(f)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.lr = 0.01
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:6])
        self.decoder = torch.nn.Sequential(    
            torch.nn.Conv2d(128*12, 128*2, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128*2, 128, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 24, kernel_size=3, stride=1, padding=1),
        )
        self.train_resnet = True
    
    def forward(self, X):
        encoder_output = torch.Tensor([])
        for sample in X:
            sample = sample.view(12, 1, 128, 128)
            sample = preprocess_sample(sample)
            output = self.encoder(sample)
            output = output.view(1, 12*128, 28, 28)
            encoder_output = torch.cat((encoder_output, output))
        Y = self.decoder(encoder_output)
        return Y*0.5+0.5
    
    def predict(self, X):
        data_X = batch_data[1]
        data_X[0] = X
        with torch.no_grad():
            Y = self.forward(data_X)
        Y = Y[0]
        Y = torchvision.transforms.GaussianBlur(kernel_size=(63, 63), sigma=3)(Y)
        return Y
    