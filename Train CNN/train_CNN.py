import torch, torchvision, random
import matplotlib.pyplot as plt
from tqdm import tqdm 
from loss import MS_SSIMLoss
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on: {}".format(str(device).upper()))
if torch.cuda.is_available():
    torch.cuda.empty_cache()

resnet = torchvision.models.resnet34(pretrained=True)
#resnet.eval()

def preprocess_sample(sample):
    sample = torchvision.transforms.Resize(224)(sample)
    sample = torch.cat((sample, sample, sample), axis=1)
    sample = torchvision.transforms.Normalize((0.4302, 0.4575, 0.4539), (0.2361, 0.2347, 0.2432))(sample)
    return sample

class Climate_dataset(torch.utils.data.Dataset):
    
    def __init__(self, files, shuffle_data=True, max_chunk_size=3000):
        # Compute chunk sizes
        self.chunk_sizes = []
        for file in files:
            with open(file, 'rb') as f:
                size = len(pickle.load(f)[:max_chunk_size])
            self.chunk_sizes.append(size)

        self.chunk_number = 0
        self.chunk_index = 0
        self.files = files
        self.shuffle_data = shuffle_data
        self.max_chunk_size = max_chunk_size
        self.get_data(files[0])
    
    def reset(self):
        if self.chunk_number != 1:
            self.get_data(self.files[0])
        else:
            if self.shuffle_data:
                random.shuffle(self.data)
        self.chunk_number = 0
        self.chunk_index = 0
        

    def get_data(self, filename):
        self.data = []

        with open(filename, 'rb') as f:
            numpy_data = pickle.load(f)

        if self.shuffle_data:
            random.shuffle(numpy_data)

        numpy_data = numpy_data[:self.max_chunk_size]

        for sample in numpy_data:
            coordinates, X, Y = sample
            coordinates, X, Y = torch.from_numpy(coordinates), torch.from_numpy(X)/1023, torch.from_numpy(Y)/1023
            self.data.append([coordinates, X, Y])

    def __len__(self): 
        return sum(self.chunk_sizes)
    
    def __getitem__(self, idx):
        sample = self.data[self.chunk_index]
        coordinates, X, Y = sample

        self.chunk_index += 1
        if self.chunk_index == self.chunk_sizes[self.chunk_number]: #Finished chunk
            self.chunk_number += 1
            if self.chunk_number == len(self.chunk_sizes): #No more chunks
                self.reset()
            else: #There are more chunks
                self.chunk_index = 0
                next_file = self.files[self.chunk_number]
                self.get_data(next_file)

        return coordinates.to(device), X.to(device), Y.to(device)

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
        self.loss_fn = MS_SSIMLoss(channels=24)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train_resnet = True
    
    def forward(self, X, train=True):
        with torch.set_grad_enabled(self.train_resnet): 
            encoder_output = torch.Tensor([]).to(device)
            for sample in X:
                sample = sample.view(12, 1, 128, 128)
                sample = preprocess_sample(sample)
                output = self.encoder(sample)
                output = output.view(1, 12*128, 28, 28)
                encoder_output = torch.cat((encoder_output, output))
        #if train:
        #  encoder_output = torch.nn.Dropout2d(0.5)(encoder_output)
        Y = self.decoder(encoder_output)
        return Y*0.5+0.5

    def trian_batch(self, X, Y):    
        pred_Y = self.forward(X)
        loss = self.loss_fn(pred_Y.unsqueeze(dim=2), Y.unsqueeze(dim=2).to(device))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss)
    
    def predict(self, X):
        with torch.no_grad():
            Y = self.forward(X, train=False)
        return Y
    
    def get_loss(self, X, Y):
        pred_Y = self.predict(X)
        loss = self.loss_fn(pred_Y.unsqueeze(dim=2), Y.unsqueeze(dim=2).to(device))
        return float(loss)

model = Model().to(device)
#model.load_state_dict(torch.load('[insert your model path]'))
print("Model loaded.")

# Put the path of your data chunks here
val_files, train_files = ['Climatehack/train10k.pickle'], ['Climatehack/train8000k.pickle', 'Climatehack/train6000k.pickle', 'Climatehack/train100k.pickle', 'Climatehack/train2000k.pickle', 'Climatehack/train7000k.pickle', 'Climatehack/train4000k.pickle']


ch_dataset = Climate_dataset(train_files) 
ch_dataloader = torch.utils.data.DataLoader(ch_dataset, batch_size=16, shuffle=False)

ch_dataset_val = Climate_dataset(val_files) 
ch_dataloader_val = torch.utils.data.DataLoader(ch_dataset_val, batch_size=16, shuffle=False)

train_size, val_size = len(ch_dataset), len(ch_dataset_val)

print("Data loaded and ready.")

model.train_resnet = False
epochs = 1000
min_val_loss = 2
min_loss = 2
max_batches = 20
print("Dataset size: "+str(train_size))

# 18.6s per batch 
for n in range(epochs):
    losses = []
    for i, (_, X, Y) in tqdm(enumerate(ch_dataloader), total=train_size//16+1, desc="Epoch "+str(n+1)):
        loss = model.trian_batch(X, Y) 
        losses.append(loss)
        if i%25 == 0:
          torch.cuda.empty_cache()
        #if i == max_batches:
        #    break
    # Val Metrics 
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    val_losses = []
    for _, val_X, val_Y in ch_dataloader_val:
        val_loss = model.get_loss(val_X, val_Y)
        val_losses.append(val_loss)
        torch.cuda.empty_cache()
    val_loss = sum(val_losses)/len(val_losses)
    avg_loss = sum(losses)/len(losses)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_train.pth')
        print("Saved best val!")
    if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save(model.state_dict(), 'best_model_val.pth')
        print("Saved best train!")
    if n%2 == 1: #Train Resnet 1 out of 2 epochs.
      print("Training ResNet.")
      model.train_resnet = True
    else:
      model.train_resnet = False
   

    print("Epoch number: {} == Loss: {} == Val Loss: {} == Min Val Loss: {}".format(n+1, avg_loss, val_loss, min_val_loss))
