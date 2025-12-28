import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import requests
import io
import numpy as np

class VisualAdversary:
    def __init__(self):
        print("[*] Initializing Visual Adversary (Surrogate Model: ResNet18)...")
        # We use a surrogate model to generate gradients (Transfer Attack)
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.criterion = nn.CrossEntropyLoss()

    def load_image(self, image_path):
        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")
        return image

    def fgsm_attack(self, image, epsilon):
        """
        Performs Fast Gradient Sign Method (FGSM) attack.
        """
        # Create input tensor directly from the image, but we need it to requires_grad
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor.requires_grad = True

        # Forward pass to get prediction
        output = self.model(input_tensor)
        
        # Get the predicted class (we want to move AWAY from this max)
        init_pred = output.max(1, keepdim=True)[1] 
        
        # Calculate loss
        loss = self.criterion(output, init_pred[0])
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect data_grad
        data_grad = input_tensor.grad.data
        
        # standard FGSM: perturbed_image = image + epsilon * sign(data_grad)
        sign_data_grad = data_grad.sign()
        perturbed_image = input_tensor + epsilon * sign_data_grad
        
        # Clipping to maintain valid image range (normalized) is complex because of 
        # standard normalization (mean/std). 
        # For simplicity in this demo, we return the tensor and will inverse-transform later 
        # or just save it. 
        
        return perturbed_image, init_pred.item()

    def generate_adversarial_example(self, image_path, epsilon=0.1):
        """
        Generates an adversarial image that fools the local surrogate model.
        Returns the binary bytes of the new image.
        """
        original_image = self.load_image(image_path)
        perturbed_tensor, original_class_id = self.fgsm_attack(original_image, epsilon)
        
        # Check if it fooled the surrogate
        output = self.model(perturbed_tensor)
        final_pred = output.max(1, keepdim=True)[1]
        
        print(f"[-] Original Class: {original_class_id}")
        print(f"[-] Perturbed Class: {final_pred.item()}")
        
        if original_class_id != final_pred.item():
            print("[+] Surrogate Model Fooled!")
        else:
            print("[!] Surrogate Model NOT fooled (increase epsilon).")

        # Convert tensor back to image bytes
        # We need to un-normalize to get visually viewable image, 
        # but for feeding to API, we just need valid image format.
        # Quick hack: Save the tensor as image. 
        # Note: This simple reversal might be lossy.
        
        # Inverse normalize (approximate for visualization/saving)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        x = perturbed_tensor.squeeze(0).detach()
        x = x * std + mean
        x = torch.clamp(x, 0, 1)
        
        to_pil = transforms.ToPILImage()
        adv_image = to_pil(x)
        
        img_byte_arr = io.BytesIO()
        adv_image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
