# Wildlife Image Classification

## Overview

This project focuses on classifying wildlife species from images using a Convolutional Neural Network (CNN). The dataset consists of 20 different classes of animals found in Oregon. A pre-trained VGG16 model is fine-tuned for transfer learning, and a custom CNN model is also implemented for classification. The project also includes a Streamlit-based web app for easy user interaction.

## Dataset

The dataset consists of **14,013 images** across **20 classes**:

- Bald Eagle (748 images)
- Black Bear (718 images)
- Cougar (680 images)
- Elk (660 images)
- Gray Wolf (730 images)
- Mountain Beaver (577 images)
- Bobcat (696 images)
- Nutria (701 images)
- Coyote (736 images)
- Columbian Black-Tailed Deer (735 images)
- Seals (698 images)
- Canada Lynx (717 images)
- Ringtail (588 images)
- Red Fox (759 images)
- Gray Fox (668 images)
- Virginia Opossum (728 images)
- Sea Lions (726 images)
- Raccoon (728 images)
- Raven (656 images)
- Deer (764 images)

## Model Architecture

The CNN model is designed to classify wildlife images efficiently. It consists of:

- **5 Convolutional Layers** with increasing filter sizes:
  - `Conv2D` filters: **16, 32, 64, 128, 256**
- **Batch Normalization** applied after each convolutional layer to stabilize training
- **MaxPooling** layers for feature extraction and dimensionality reduction
- **Dropout Layers** to prevent overfitting and improve generalization
- **Fully Connected Layers** for final classification

### Loss & Optimization

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam`

## Model Training

### Transfer Learning using VGG16

A pre-trained VGG16 model is used, with its feature extraction layers frozen and a custom fully connected layer added for classification.

```python
import torchvision.models as models
import torch.nn as nn

model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.requires_grad = False

n_inputs = model_transfer.classifier[6].in_features
last_layer = nn.Linear(n_inputs, 133)
model_transfer.classifier[6] = last_layer

if use_cuda:
    model_transfer.cuda()
```

### Custom CNN Model

A custom CNN model is also implemented with convolutional, batch normalization, dropout, and fully connected layers.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 20)

    def forward(self, x):
        x = self.conv_bn2(self.pool(F.relu(self.conv1(x))))
        x = self.conv_bn3(self.pool(F.relu(self.conv2(x))))
        x = self.conv_bn4(self.pool(F.relu(self.conv3(x))))
        x = self.conv_bn5(self.pool(F.relu(self.conv4(x))))
        x = self.conv_bn6(self.pool(F.relu(self.conv5(x))))
        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## Streamlit Web Application

A **Streamlit** application is built to allow users to upload images and classify them using the trained model.

### Features

- Upload an image and get the predicted wildlife species.
- Uses **custom CNN model**.
- GPU acceleration (if available).

### Running the App

Install dependencies:

```bash
pip install numpy pandas torch torchvision matplotlib pillow streamlit kagglehub
```

Run the app:

```bash
streamlit run app.py
```

### Streamlit Code Overview

```python
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

def predict_breed_transfer(img_path, model, class_names, use_cuda):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    if use_cuda:
        img_tensor = img_tensor.cuda()
    model.eval()
    output = model(img_tensor)
    predict_index = output.cpu().data.numpy().argmax()
    return class_names[predict_index]

st.title("Wildlife Image Classifier")
img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if img_file is not None:
    st.image(img_file, caption="Uploaded Image", use_column_width=True)
    use_cuda = torch.cuda.is_available()
    model = Net()
    model.load_state_dict(torch.load('model_scratch.pt', map_location='cpu'))
    class_names = ["Bald Eagle", "Black Bear", "Cougar", "Elk", "Gray Wolf", ...]
    predicted_breed = predict_breed_transfer(img_file, model, class_names, use_cuda)
    st.write(f"Predicted Animal: {predicted_breed}")
```

## File Structure

```
|-- app.py  # Streamlit application
|-- model_scratch.pt  # Trained model weights
|-- Wildlife_Image_Classification.ipynb # Colab Notebook
|-- README.md  # Project documentation
```

## Conclusion

This project successfully implements a wildlife image classification model using both transfer learning and a custom CNN. The Streamlit web app provides a user-friendly interface to classify images in real time.
