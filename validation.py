import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score
import os
import platform
from vit_model import ViT_class  # Import your ViT model definition
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import torch
from torch import nn
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
 # Create a mapping of unique labels to integers
        self.label_to_int = {label: i for i, label in enumerate(dataframe['label'].unique())}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}
        # Base transformation: Normalization, resize, and any additional basic transformations
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
        ])

        # Augmentation: Random rotation and adding noise
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))  # Adding Gaussian noise
        ])

    def __len__(self):
        return len(self.dataframe)

    def load_and_transform_image(self, img_path, apply_augmentation=False):
        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Apply base transform and augmentation
        if apply_augmentation:
            image_tensor = self.augmentation(image)
        else:
            image_tensor = self.base_transform(image)

        return image_tensor

    def __getitem__(self, idx):
        # Get image path and label
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = row['label']

        # Load and transform image
        image = self.load_and_transform_image(img_path, apply_augmentation=self.transform)

        # Convert label to integer
        label_int = self.label_to_int[label]

        # Convert label to tensor
        label_tensor = torch.tensor(label_int)

        return image, label_tensor

    def get_num_classes(self):
        return len(self.label_to_int)

    def get_class_names(self):
        return list(self.label_to_int.keys())

def load_and_preprocess_image(full_path, transform):
    """Load and preprocess a single image"""
    img = Image.open(full_path).convert('RGB')
    return transform(img)

def get_data(excel_path, base_dir, transform):
    """Load and preprocess validation data"""
    df = pd.read_excel(excel_path)
    df = df.dropna(subset=['image_path'])
    
    # Handle path separators based on OS
    if platform.system() == 'Windows':
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('/', os.sep))
    else:
        df['image_path'] = df['image_path'].apply(lambda x: x.replace('\\', os.sep))
    
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
                    'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
    # Load and preprocess images
    X = torch.stack([
        load_and_preprocess_image(os.path.join(base_dir, path), transform) 
        for path in df['image_path'].values
    ])
    
    y = df[class_columns].values
    return X, y, df

def load_test_data(test_dir, transform):
    """Load and preprocess test data"""
    image_filenames = [fname for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    X_test = torch.stack([
        load_and_preprocess_image(os.path.join(test_dir, fname), transform) 
        for fname in image_filenames
    ])
    return X_test, image_filenames


    # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model configuration
config = {
    "patch_size": 8,
    "hidden_size": 768//2,
    "num_hidden_layers": 6,
    "num_attention_heads": 6,
    "intermediate_size": 4 * 384,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 224,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

# Initialize model
model = ViT_class(config)
model.load_state_dict(torch.load('best_vit_model.pth'))
model = model.to(device)
model.eval()


# Paths
root_dir = 'Dataset/validation'
data=[]
# Traverse through all directories and files
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.jpg'):  # Check for image files
            full_path = os.path.join(subdir, file)
            # Extract the label from the folder name (first folder)
            label = full_path.split(os.sep)[-3]  # Assuming structure is label/_/image.jpg
            data.append([full_path, label])

# Create the DataFrame
val_df = pd.DataFrame(data, columns=['path', 'label'])   
    # Define transforms
batch_size = 16
val_dataset = CustomImageDataset(val_df)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validation"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

# Generate metrics report
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
results_df = pd.DataFrame()

# Define class names explicitly
class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
               'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Add probability columns for each class
for idx, class_name in enumerate(class_names):
    results_df[class_name] = [probs[idx] for probs in all_probs]

# Add predicted class name
results_df['predicted_class'] = [class_names[pred] for pred in all_preds]

# Save to Excel
results_df.to_excel('validation_results.xlsx', index=False)