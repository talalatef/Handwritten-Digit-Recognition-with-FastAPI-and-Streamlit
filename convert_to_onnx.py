# convert_to_onnx.py
import torch
import torch.onnx
from train_model import SimpleCNN

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Dummy input for ONNX conversion
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model
torch.onnx.export(model, dummy_input, "mnist_cnn.onnx", verbose=True, input_names=['input'], output_names=['output'])
