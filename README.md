# Image Captioning Project

This project aims to generate meaningful captions for images using deep learning models. Three different architectures are used for this task: VGG16 + LSTM, ResNet + LSTM, and InceptionV3 + Attention + LSTM.

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Prediction](#prediction)
- [Streamlit Application](#streamlit-application)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Introduction

Image captioning is the process of generating textual descriptions for images. This project leverages three different deep learning models to perform this task:

1. **VGG16 + LSTM**: Uses VGG16 for feature extraction and LSTM for sequential data handling.
2. **ResNet + LSTM**: Utilizes ResNet for deep feature extraction and LSTM for caption generation.
3. **InceptionV3 + Attention + LSTM**: Combines InceptionV3 for efficient feature extraction, attention mechanism for focusing on relevant parts of the image, and LSTM for generating captions.

## Data Preparation

Images are preprocessed using standard techniques such as resizing, normalization, and data augmentation (e.g., rotation, flipping, zooming) to improve model robustness. Captions are tokenized and converted to sequences of word indices, with padding applied to ensure consistent sequence lengths.

## Model Architectures

### VGG16 + LSTM

- **Feature Extraction**: VGG16 is used to extract high-level features from images.
- **Sequential Data Handling**: An LSTM network processes the extracted features to generate coherent captions.
- **Optimization**: Fine-tuning and batch normalization are employed to enhance performance.

### ResNet + LSTM

- **Deep Architecture**: ResNet addresses the vanishing gradient problem, enabling effective training of deep networks.
- **Sequential Processing**: LSTM handles caption generation, focusing on sequential and contextually relevant descriptions.
- **Enhancements**: Skip connections between ResNet and LSTM for enhanced context understanding.

### InceptionV3 + Attention + LSTM

- **Efficient Feature Extraction**: InceptionV3 utilizes multiple kernel sizes for capturing features at various scales.
- **Attention Mechanism**: Directs focus to relevant image parts, enhancing the precision of generated captions.
- **Dynamic Training Parameters**: Continuously optimized attention weights during validation for relevance and accuracy.

## Training

- **Loss Function**: Cross-entropy loss is used to measure the difference between the predicted and actual captions.
- **Optimization**: The Adam optimizer is employed for its adaptive learning rate capabilities.
- **Early Stopping**: Implemented to prevent overfitting by monitoring validation loss and stopping training when performance no longer improved.
- **Mini-Batch Training**: Used to efficiently handle large datasets and improve training stability.

## Prediction

The trained models can be used to generate captions for new images. The `predict.py` script handles the prediction process using the specified model and image path.

## Streamlit Application

A Streamlit application has been developed for easy interaction with the trained models. This application allows users to upload an image and generate captions using the trained models.

### Running the Streamlit Application

1. Install Streamlit if you haven't already:
    ```bash
    pip install streamlit
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Open your browser and navigate to `http://localhost:8501` to use the application.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- pandas
- spacy
- nltk
- PIL
- argparse
- streamlit

## Usage

### Data Preparation

Place the dataset in the specified directory structure:
/data/flicker8k
|-- Images/
|-- captions.txt


### Training

Train the models using the following commands:

```bash
python main.py train --model vgg16 --data_path /path/to/data --captions_file /path/to/captions.txt --batch_size 64 --embed_size 256 --hidden_size 512 --num_layers 1 --num_epochs 5
python main.py train --model resnet --data_path /path/to/data --captions_file /path/to/captions.txt --batch_size 64 --embed_size 256 --hidden_size 512 --num_layers 1 --num_epochs 5
python main.py train --model inception --data_path /path/to/data --captions_file /path/to/captions.txt --batch_size 64 --embed_size 256 --hidden_size 512 --num_layers 1 --num_epochs 5
```
Generate captions using the following commands:

```bash
python predict.py --model vgg16 --model_path /path/to/vgg16_model.pth --image_path /path/to/image.jpg --data_path /path/to/data --captions_file /path/to/captions.txt
python predict.py --model resnet --model_path /path/to/resnet_model.pth --image_path /path/to/image.jpg --data_path /path/to/data --captions_file /path/to/captions.txt
python predict.py --model inception --model_path /path/to/inception_model.pth --image_path /path/to/image.jpg --data_path /path/to/data --captions_file /path/to/captions.txt
```


<img width="1242" alt="image" src="https://github.com/AbhinanduReddy/image-captioning/assets/43111492/e62091e8-009e-4ff0-8a95-4b6f5d5b7179">
