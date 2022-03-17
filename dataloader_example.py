import torch, random
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on: {}".format(str(device).upper()))

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

# Put the path of your data chunks here
# Download the files from the Google drive link.
# It can take a few seconds to load chunks
train_files = ['train8000k.pickle', 'train6000k.pickle', 'train100k.pickle']


ch_dataset = Climate_dataset(train_files) 
ch_dataloader = torch.utils.data.DataLoader(ch_dataset, batch_size=16, shuffle=False)

train_size = len(ch_dataset)
print("Dataset size: "+str(train_size))

for coordinates, X, Y in ch_dataloader: #Run 1 epoch
    #[Insert great model here.]
    print(coordinates.shape, X.shape, Y.shape)

