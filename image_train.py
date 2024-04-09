import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '/data/cmpe258-sp24/flicker8k'
print('Folders present in the data', os.listdir(data_path))


captions_path = data_path + '/captions.txt'

captions_data =  pd.read_csv(captions_path)

captions_data.head(5)
# Define dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer or word_tokenize
        
        # Load image paths and captions from captions file
        self.image_paths, self.captions = self.load_captions(captions_file)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #print('index',idx)
        #print(self.image_paths[idx])
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        caption = self.captions[idx]
        tokens = self.tokenize(caption)
        return image, tokens
    
    def load_captions(self, captions_file):
        df = pd.read_csv(captions_file)  # Read captions file using pandas
        image_paths = df['image'].tolist()
        captions = df['caption'].tolist()
        return image_paths, captions
    
    def tokenize(self, text):
        return self.tokenizer(text.lower())
    def collate_fn(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = rnn_utils.pad_sequence([torch.Tensor([vocab(token) for token in caption[0]]) for caption in captions], batch_first=True)
        return images, captions



# Define CNN+LSTM model
# class CNNLSTMModel(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(CNNLSTMModel, self).__init__()
#         self.resnet = models.resnet50(pretrained=True)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)  # Update input size
#         self.linear = nn.Linear(hidden_size, vocab_size)
        
#     def forward(self, images, captions):
#         features = self.resnet(images)
#         features = features.view(features.size(0), -1)  # Flatten features
#         features = features.unsqueeze(1).expand(-1, captions.size(1), -1)  # Expand features
#         embeddings = self.embed(captions)
#         inputs = torch.cat((features, embeddings), dim=2)  # Concatenate along dimension 2
#         hiddens, _ = self.lstm(inputs)
#         outputs = self.linear(hiddens)
#         return outputs

class CNNLSTMModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # Flatten features
        features = features.unsqueeze(1).expand(-1, captions.size(1), -1)  # Expand features
        embeddings = self.embed(captions)
        inputs = torch.cat((features, embeddings), dim=2)  # Concatenate along dimension 2
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, image, max_len=20):
        """
        Generate captions for given image features using greedy decoding.
        """
        result_caption = []
        with torch.no_grad():
            # Encode image features
            features = self.resnet(image)
            
            # Get original size of the features tensor
            original_size = features.size()

            # Handle cases where the batch size is 1 (original_size[0] is 1)
            if original_size[0] == 1:
                fourth_dim_size = 2048  # Default value if batch size is 1
            else:
                fourth_dim_size = int(original_size[0] * original_size[1] * original_size[2] * original_size[3] / (max_len * original_size[4]))

            # Reshape features for better clarity
            features = features.view(features.size(0), 1, 1, fourth_dim_size, max_len, features.size(1))

            # Initial input to the LSTM (start token)
            inputs = torch.tensor([[vocab('<start>')]], device=device)
            for _ in range(max_len):
                # Embedding for the current input token
                embeddings = self.embed(inputs)
                # Concatenate image features and embeddings
                inputs = torch.cat((features, embeddings.unsqueeze(2)), dim=3)  # Concatenate along dimension 4
                # Forward pass through LSTM
                hiddens, _ = self.lstm(inputs)
                # Predict next token
                outputs = self.linear(hiddens.squeeze(1))
                _, predicted = outputs.max(2)
                # Append predicted token to result caption
                result_caption.append(predicted.item())
                # If end token is predicted, stop generating
                if predicted.item() == vocab('<end>'):
                    break
                # Prepare input for next iteration
                inputs = embeddings.new_tensor([[predicted.item()]])
        return result_caption

# Function to train the model
def train_model(model, criterion, optimizer, dataloader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device).long()  # Convert captions tensor to LongTensor
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images, captions[:, :-1])
            # Calculate loss
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss/len(dataloader)))

# Function to predict captions
def predict_caption(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        sampled_ids = model.sample(image)
        sampled_ids = sampled_ids[0].cpu().numpy()
        predicted_caption = ' '.join([vocab.idx2word[id] for id in sampled_ids if id not in [vocab('<start>'), vocab('<end>')]])
        return predicted_caption

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalization parameters for ImageNet
])

# Path to dataset
# data_path = '/path/to/dataset'
# captions_path = '/path/to/captions.csv'

# Load dataset
dataset = Flickr8kDataset(data_path+'/Images', captions_path, transform=transform)
import random
import numpy as np
idx = random.randint(0, len(dataset) - 1)
image, caption = dataset[idx]
print('imagesize', image.size())
print('captions size',len(caption))

# Print the caption
print('Caption:', caption)
split_fraction = 0.3  # For example, use 10% of the dataset
total_samples = len(dataset)
subset_size = int(total_samples * split_fraction)
subset_indices = torch.randperm(total_samples)[:subset_size]
subset = torch.utils.data.Subset(dataset, subset_indices)

# Split dataset into train and validation sets
train_set, val_set = train_test_split(subset, test_size=0.1, random_state=42)
print('datadon')
# Define data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)

# Vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<unk>')  # Add <unk> token

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']  # Return index of <unk> token for unknown words
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

vocab = Vocabulary()
for _, captions in train_loader:
    for caption in captions:
        for word in caption:
            vocab.add_word(word)

# Model parameters
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)
num_layers = 1

# Initialize model, loss function, and optimizer
model = CNNLSTMModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, criterion, optimizer, train_loader)

# # Example prediction
# image, _ = val_set[0]
# predicted_caption = predict_caption(model, image.unsqueeze(0))
# print('Predicted caption:', predicted_caption)
