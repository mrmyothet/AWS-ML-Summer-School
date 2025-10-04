#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install requirements
get_ipython().system('pip install facenet-pytorch')


# In[1]:


import torch
import numpy as np
import random
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt


# In[2]:


# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Load test data
path = "CelebA_test_images/images"
print("Path to dataset files:", path)
all_faces=glob.glob(path+"/*.jpg")
data_size=len(all_faces)
print(f"Total images : {data_size}")


# In[3]:


device


# In[4]:


# load a subset of test images

def get_face_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path)
    # resize
    face = transform(img)
    return face

seed = 42
np.random.seed(seed)
all_faces = np.array(all_faces)
np.random.shuffle(all_faces)

# change n to increase or decrease the size of the test set
n = 2000
all_faces_test = all_faces[:n]

image_tensor_test=[]

for x in all_faces_test:
  image_tensor_test.append(get_face_image(x))


# In[5]:


def visualize_trigger(trigger):
    """
    Visualize a learned or fixed trigger (grayscale or RGB).

    Args:
        trigger (torch.Tensor): Trigger tensor, shape (C, H, W) or (H, W).
    """
    trigger_np = trigger.squeeze().detach().cpu().numpy()

    # Print rounded values
    print(np.round(trigger_np, 2))

    # Handle grayscale vs RGB
    if trigger_np.ndim == 2:  # (H, W)
        plt.imshow(trigger_np, cmap='gray')
    elif trigger_np.ndim == 3:  # (C, H, W)
        # Convert (C,H,W) → (H,W,C)
        trigger_np = np.transpose(trigger_np, (1, 2, 0))
        plt.imshow(trigger_np)
    else:
        raise ValueError(f"Unexpected trigger shape: {trigger_np.shape}")

    plt.colorbar()
    plt.show()


# In[6]:


class PoisonedTestSet(Dataset):
    def __init__(self, base_dataset, target_label, trigger_func):
        """
        base_dataset: the clean dataset to poison
        target_label: the label assigned to all poisoned samples
        trigger_func: function to inject the trigger (default = add_trigger_func1)
        """
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.trigger_func = trigger_func

    def __getitem__(self, idx):
        img = self.base_dataset[idx]
        img = self.trigger_func(img)  # Apply trigger
        return img, self.target_label

    def __len__(self):
        return len(self.base_dataset)


# In[7]:


def attack_success_rate(Model, loader):
    Model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = Model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f"Accuracy: {100*correct/total:.2f}%")


# In[24]:


def apply_trigger_opt(images, trigger_patch,trigger_size):
    patched = images.clone()
    patched[:, :, -trigger_size:, -trigger_size:] = trigger_patch
    return patched

def optimize_trigger_for_asr(model, base_dataset, target_label=0, steps=100, lr=0.1, trigger_size=20):
    model.eval()
    loader = DataLoader(base_dataset, batch_size=64, shuffle=False)

    # Trainable trigger
    trigger = torch.randn((1, 3, trigger_size, trigger_size), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([trigger], lr=lr)

    #TODO
    criterion = nn.CrossEntropyLoss()
    
    print(f"Optimizing trigger for {steps} steps...")
    
    for step in range(steps):
        total_loss = 0.0
        num_batches = 0
        
        for batch_images in loader:
            if isinstance(batch_images, list):
                batch_images = torch.stack(batch_images)
            
            batch_images = batch_images.to(device)
            batch_size = batch_images.size(0)
            
            triggered_images = apply_trigger_opt(batch_images, trigger, trigger_size)
            
            targets = torch.full((batch_size,), target_label, dtype=torch.long, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(triggered_images)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (step + 1) % 1 == 0 or step == 0:
            avg_loss = total_loss / num_batches
            print(f"Step {step + 1}/{steps}, Average Loss: {avg_loss:.4f}")
    
    with torch.no_grad():
        trigger.clamp_(0, 1)

    
    return trigger


# In[25]:


#load the model
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=1000)
model = torch.nn.DataParallel(model).to(device)
ckp = torch.load("model_weights_poisoned_partC_facenet2.tar", map_location=device)
model.load_state_dict(ckp['state_dict'])
model.eval()


# In[26]:


# Optimize trigger
optimized_trigger = optimize_trigger_for_asr(model, image_tensor_test, target_label=0,steps=10, lr=0.5)
# optimized_trigger = torch.randn((1, 3, 20, 20), requires_grad=True) # Fill it with the trigger value you have obtained

# Print and visualize trigger
visualize_trigger(optimized_trigger)

# Use this trigger
print(optimized_trigger)


# In[27]:


def add_trigger_opt(images, trigger_patch = optimized_trigger, trigger_size = 20):
    patched = images.clone()
    patched[:, -trigger_size:, -trigger_size:] = trigger_patch
    return patched

test_set_poisoned_opt = PoisonedTestSet(image_tensor_test, trigger_func=add_trigger_opt, target_label=0)
test_loader_opt_triggered = DataLoader(test_set_poisoned_opt, batch_size=1000, shuffle=False)

# Report this ASR
attack_success_rate(model,test_loader_opt_triggered)


# In[10]:


# Optimize trigger
optimized_trigger = optimize_trigger_for_asr(model, image_tensor_test, target_label=0,steps=1, lr=0.01)
# optimized_trigger = torch.randn((1, 3, 20, 20), requires_grad=True) # Fill it with the trigger value you have obtained

# Print and visualize trigger
visualize_trigger(optimized_trigger)

# Use this trigger
print(optimized_trigger)


# In[11]:


def add_trigger_opt(images, trigger_patch = optimized_trigger, trigger_size = 20):
    patched = images.clone()
    patched[:, -trigger_size:, -trigger_size:] = trigger_patch
    return patched

test_set_poisoned_opt = PoisonedTestSet(image_tensor_test, trigger_func=add_trigger_opt, target_label=0)
test_loader_opt_triggered = DataLoader(test_set_poisoned_opt, batch_size=1000, shuffle=False)

# Report this ASR
attack_success_rate(model,test_loader_opt_triggered)


# In[ ]:




