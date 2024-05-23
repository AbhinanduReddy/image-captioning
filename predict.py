import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os
from models import VGG16LSTM, ResNetLSTM, InceptionV3AttentionLSTM
from data import Vocabulary, Flickr8kDataset

# Argument parser
parser = argparse.ArgumentParser(description='Predict image captions using trained models')
parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet', 'inception'], help='Model type')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--image_path', type=str, required=True, help='Path to the image for prediction')
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--captions_file', type=str, required=True, help='Path to the captions file')

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = Flickr8kDataset(args.data_path, args.captions_file, transform=transform)
vocab = dataset.vocab

# Load the model
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)
num_layers = 1

if args.model == 'vgg16':
    model = VGG16LSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
elif args.model == 'resnet':
    model = ResNetLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)
elif args.model == 'inception':
    model = InceptionV3AttentionLSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Load and preprocess the image
image = Image.open(args.image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict caption
with torch.no_grad():
    if args.model == 'inception':
        features = model.inception(image)
        features = features.view(features.size(0), -1)
        predicted_caption = model.sample(features.unsqueeze(0), vocab)
    else:
        predicted_caption = model.sample(image, vocab)

predicted_caption = ' '.join([vocab.itos[word] for word in predicted_caption if word not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])
print(f'Predicted Caption: {predicted_caption}')
