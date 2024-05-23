import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data_loaders
from models import VGG16LSTM, ResNetLSTM, InceptionV3AttentionLSTM
from nltk.translate.bleu_score import sentence_bleu

# Define argument parser
parser = argparse.ArgumentParser(description='Image Captioning Training and Prediction')
parser.add_argument('mode', type=str, choices=['train', 'predict'], help='Mode: train or predict')
parser.add_argument('--model', type=str, choices=['vgg16', 'resnet', 'inception'], help='Model type')
parser.add_argument('--data_path', type=str, default='/data/cmpe258-sp24/flicker8k', help='Path to the data')
parser.add_argument('--captions_file', type=str, default='/data/cmpe258-sp24/flicker8k/captions.txt', help='Path to the captions file')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--embed_size', type=int, default=256, help='Embedding size')
parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--image_path', type=str, help='Path to the image for prediction')

args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_loader, val_loader, vocab = get_data_loaders(args.data_path, args.captions_file, args.batch_size)

# Initialize model
if args.model == 'vgg16':
    model = VGG16LSTM(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
elif args.model == 'resnet':
    model = ResNetLSTM(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
elif args.model == 'inception':
    model = InceptionV3AttentionLSTM(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, dataloader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device).long()
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1])
            loss = criterion(outputs.view(-1, len(vocab)), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            for i in range(len(images)):
                predicted_caption = model.sample(images[i].unsqueeze(0), vocab)
                predicted_caption = ' '.join([vocab.itos[word] for word in predicted_caption if word not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])
                reference_caption = ' '.join([vocab.itos[word.item()] for word in captions[i] if word.item() not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])
                bleu_scores.append(sentence_bleu([reference_caption.split()], predicted_caption.split()))
    return sum(bleu_scores) / len(bleu_scores)

if args.mode == 'train':
    print(f'Training {args.model.upper()} Model')
    train_model(model, optimizer, train_loader, args.num_epochs)
    print(f'Evaluating {args.model.upper()} Model')
    bleu_score = evaluate_model(model, val_loader)
    print(f'BLEU Score: {bleu_score:.4f}')

elif args.mode == 'predict':
    if not args.image_path:
        raise ValueError('Please provide an image path for prediction')
    from PIL import Image
    from torchvision import transforms
    image = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).to(device)
    predicted_caption = model.sample(image, vocab)
    predicted_caption = ' '.join([vocab.itos[word] for word in predicted_caption if word not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]]])
    print(f'Predicted Caption: {predicted_caption}')
