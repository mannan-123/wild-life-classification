import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import re

# Define the CNN model (same as before)


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

        self.conv_bn1 = nn.BatchNorm2d(224, 3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 20)

    def forward(self, x):
        x = self.conv_bn2(self.pool(F.relu(self.conv1(x)))
                          )  # Use F.relu for activation
        x = self.conv_bn3(self.pool(F.relu(self.conv2(x)))
                          )  # Use F.relu for activation
        x = self.conv_bn4(self.pool(F.relu(self.conv3(x)))
                          )  # Use F.relu for activation
        x = self.conv_bn5(self.pool(F.relu(self.conv4(x)))
                          )  # Use F.relu for activation
        x = self.conv_bn6(self.pool(F.relu(self.conv5(x)))
                          )  # Use F.relu for activation
        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  # Use F.relu for activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the inference function


def predict_breed_transfer(img_path, model, class_names, use_cuda):
    img = Image.open(img_path)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img).float()
    img_tensor.unsqueeze_(0)
    img_tensor = Variable(img_tensor)

    if use_cuda:
        img_tensor = img_tensor.cuda()

    model.eval()
    output = model(img_tensor)
    output = output.cpu()

    predict_index = output.data.numpy().argmax()
    predicted_breed = class_names[predict_index]

    return predicted_breed

# Streamlit interface


def main():
    st.title("Animal Prediction")
    st.write("Upload an image to get the predicted animal.")

    # Upload image
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        # Display uploaded image
        st.image(img_file, caption="Uploaded Image", use_column_width=True)

        # Check if CUDA is available
        use_cuda = torch.cuda.is_available()

        # Initialize the model
        model_scratch = Net()
        if use_cuda:
            model_scratch.cuda()

        # Load pre-trained model
        model_scratch.load_state_dict(torch.load(
            'model_scratch.pt', map_location=torch.device('cpu')))

        # Class names (adjust accordingly to match the model classes)
        class_names = ['Bald Eagle', 'Black Bear', 'Bobcat', 'Canada Lynx', 'Columbian Black-Tailed Deer', 'Cougar', 'Coyote', 'Deer', 'Elk',
                       'Gray Fox', 'Gray Wolf', 'Mountain Beaver', 'Nutria', 'Raccoon', 'Raven', 'Red Fox', 'Ringtail', 'Sea Lions', 'Seals', 'Virginia Opossum']

        # Perform prediction
        predicted_breed = predict_breed_transfer(
            img_file, model_scratch, class_names, use_cuda)

        # Display prediction result
        st.write(f"Predicted Animal: {predicted_breed}")


if __name__ == '__main__':
    main()
