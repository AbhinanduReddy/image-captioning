import torch
import torch.nn as nn
import torchvision.models as models

class VGG16LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(VGG16LSTM, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 4096, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.vgg(images)
        features = features.unsqueeze(1).expand(-1, captions.size(1), -1)
        embeddings = self.embed(captions)
        inputs = torch.cat((features, embeddings), dim=2)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, image, vocab, max_len=20):
        result_caption = []
        with torch.no_grad():
            features = self.vgg(image)
            features = features.unsqueeze(1)
            inputs = torch.zeros((1, 1, 4096 + embed_size)).to(image.device)
            for _ in range(max_len):
                inputs[:, :, :4096] = features
                hiddens, _ = self.lstm(inputs)
                outputs = self.linear(hiddens.squeeze(1))
                _, predicted = outputs.max(1)
                result_caption.append(predicted.item())
                if predicted.item() == vocab.stoi["<EOS>"]:
                    break
                inputs[:, :, 4096:] = self.embed(predicted).unsqueeze(1)
        return result_caption

class ResNetLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ResNetLSTM, self).__init__()
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
    
    def sample(self, image, vocab, max_len=20):
        result_caption = []
        with torch.no_grad():
            features = self.resnet(image)
            features = features.view(features.size(0), -1)  # Flatten features
            features = features.unsqueeze(1)  # Add time dimension
            inputs = torch.zeros((1, 1, 2048 + embed_size)).to(image.device)
            for _ in range(max_len):
                inputs[:, :, :2048] = features
                hiddens, _ = self.lstm(inputs)
                outputs = self.linear(hiddens.squeeze(1))
                _, predicted = outputs.max(1)
                result_caption.append(predicted.item())
                if predicted.item() == vocab.stoi["<EOS>"]:
                    break
                inputs[:, :, 2048:] = self.embed(predicted).unsqueeze(1)
        return result_caption

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
    
    def forward(self, features, hidden):
        hidden = hidden.unsqueeze(1).expand_as(features)
        combined = torch.cat((features, hidden), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        attention_weighted_encoding = (features * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class InceptionV3AttentionLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(InceptionV3AttentionLSTM, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        self.inception = nn.Sequential(*list(self.inception.children())[:-1])
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(2048, hidden_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.inception(images)
        features = features.view(features.size(0), -1)  # Flatten features
        embeddings = self.embed(captions)
        lstm_input = torch.cat((features.unsqueeze(1).expand(-1, captions.size(1), -1), embeddings), dim=2)
        hiddens, _ = self.lstm(lstm_input)
        context, _ = self.attention(features, hiddens)
        outputs = self.linear(context)
        return outputs
    
    def sample(self, image, vocab, max_len=20):
        result_caption = []
        with torch.no_grad():
            features = self.inception(image)
            features = features.view(features.size(0), -1)  # Flatten features
            inputs = torch.zeros((1, 1, 2048 + embed_size)).to(image.device)
            for _ in range(max_len):
                context, _ = self.attention(features, inputs)
                inputs = torch.cat((features.unsqueeze(1), context.unsqueeze(1)), dim=2)
                hiddens, _ = self.lstm(inputs)
                outputs = self.linear(hiddens.squeeze(1))
                _, predicted = outputs.max(1)
                result_caption.append(predicted.item())
                if predicted.item() == vocab.stoi["<EOS>"]:
                    break
                inputs = self.embed(predicted).unsqueeze(1)
        return result_caption
