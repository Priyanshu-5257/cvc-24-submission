import os
import pandas as pd
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
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
import wandb
import bitsandbytes as bnb

# Initialize a list to store paths and labels
data = []

# Root directory containing the train dataset
root_dir = ' '

# Traverse through all directories and files
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.jpg'):  # Check for image files
            full_path = os.path.join(subdir, file)
            # Extract the label from the folder name (first folder)
            label = full_path.split(os.sep)[-3]  # Assuming structure is label/_/image.jpg
            data.append([full_path, label])

# Create the DataFrame
train_df = pd.DataFrame(data, columns=['path', 'label'])

data = []

# Root directory containing the validation dataset
root_dir = ' '

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


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False, train_student=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        all_blocks = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
            if train_student:
              all_blocks.append(x)
        # Return the encoder's output and the attention probabilities (optional)
        if output_attentions:
            return (x, all_attentions)
        if train_student:
            return (x, all_blocks)
        else :
            return (x, None)


class ViT(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        #self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False, train_student = False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_ = self.encoder(embedding_output, output_attentions=output_attentions,train_student = train_student)
        return encoder_output,all_
        # Calculate the logits, take the [CLS] token's output as features for classification
        # logits = self.classifier(encoder_output[:, 0, :])
        # # Return the logits and the attention probabilities (optional)
        # if output_attentions:
        #     return (logits, all_)
        # if train_student:
        #     return (logits, all_)
        # else :
        #     return (logits, None)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 8,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 768//2,
    "num_hidden_layers": 6,
    "num_attention_heads": 6,
    "intermediate_size": 4 * 384, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 224,
    "num_classes": 10, 
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

model = ViT(config)
model.to(device)
print(model)


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_pairs=100000, augment_prob=0.75):
        self.dataframe = dataframe
        self.transform = transform
        self.augment_prob = augment_prob
        self.label_to_ids = self._group_ids_by_label()
        self.paired_idx = self._prepare_pairs(num_pairs)
        random.shuffle(self.paired_idx)

        # Base transformation: Normalization, resize, and any additional basic transformations
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
        ])

        # Augmentation: Random rotation and adding noise
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(360),      
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))  # Adding Gaussian noise
        ])

    def __len__(self):
        return len(self.paired_idx)

    def _group_ids_by_label(self):
        label_to_ids = defaultdict(list)
        for idx, row in self.dataframe.iterrows():
            label_to_ids[row['label']].append(idx)
        return label_to_ids

    def _prepare_pairs(self, num_pairs):
        paired_idx = []
        labels = list(self.label_to_ids.keys())
        
        for _ in range(num_pairs):
            label = random.choice(labels)
            other_labels = [l for l in labels if l != label]
            
            i1, i2 = random.sample(self.label_to_ids[label], 2)
            i3 = random.choice(self.label_to_ids[random.choice(other_labels)])
            
            paired_idx.append([i1, i2, i3])
        
        return paired_idx

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
        # Retrieve image paths using the indices in paired_idx
        id1, id2, id3 = self.paired_idx[idx]
        path1 = self.dataframe.iloc[id1]['path']
        path2 = self.dataframe.iloc[id2]['path']
        path3 = self.dataframe.iloc[id3]['path']

        # Load anchor image
        anchor = self.load_and_transform_image(path1)

        # Decide whether to use augmented anchor or positive sample
        if random.random() < self.augment_prob:
            positive = self.load_and_transform_image(path1, apply_augmentation=True)  # Augmented anchor
        else:
            positive = self.load_and_transform_image(path2)  # Positive sample

        # Load negative sample
        negative = self.load_and_transform_image(path3)

        return anchor, positive, negative


num_pairs_train = 200000
num_pairs_val = 10000
train_dataset = CustomImageDataset(train_df,num_pairs=num_pairs_train)
batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CustomImageDataset(val_df,num_pairs=num_pairs_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for batch in train_loader:
    anchor, positive, negative = batch
    print(anchor.shape, positive.shape, negative.shape)
    break

for batch in val_loader:
    anchor, positive, negative = batch
    print(anchor.shape, positive.shape, negative.shape)
    break

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive, negative):
        """
        InfoNCE loss for single GPU.
        
        Args:
            query: Tensor of shape [B, E]
            positive: Tensor of shape [B, E]
            negative: Tensor of shape [B, E]
            
            where B is the batch size and E is the embedding dimension.
        """
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Compute logits
        l_pos = torch.einsum('ne,ne->n', query, positive).unsqueeze(-1)
        l_neg = torch.einsum('ne,me->nm', query, negative)

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature scaling
        logits /= self.temperature

        # Labels are always zero (positive pair is always the first)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
    
def train_vit_model(model, train_dataloader, val_dataloader, num_epochs, device, 
                    criterion, batch_size, num_pairs, learning_rate=2e-5, 
                    project_name="CVC-2024", run_name=None, note =None):
    # Initialize wandb
    wandb.init(project=project_name, name=run_name,notes=note, config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "num_pairs": num_pairs,
        "model": model.__class__.__name__,
        "optimizer": "Adam",
        "scheduler": "LinearWarmup"
    })

    model.to(device)
    
    # Optimizer and scheduler setup
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    total_steps = int(num_pairs / batch_size) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in train_bar:
            anchor, positive, negative = [b.to(device) for b in batch]

            optimizer.zero_grad()
            
            anchor_output,_ = model(anchor)
            positive_output,_ = model(positive)
            negative_output,_ = model(negative)
            anchor_output, positive_output, negative_output = anchor_output[:,0,:], positive_output[:,0,:], negative_output[:,0,:]
            loss = criterion(anchor_output, positive_output, negative_output)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
            
            # Log training loss to wandb
            wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})
            if scheduler.get_last_lr()[0] == 0:
                break
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                anchor, positive, negative = [b.to(device) for b in batch]
            

                anchor_output,_ = model(anchor)
                positive_output,_ = model(positive)
                negative_output,_ = model(negative)
                
                anchor_output, positive_output, negative_output = anchor_output[:,0,:], positive_output[:,0,:], negative_output[:,0,:]
                loss = criterion(anchor_output, positive_output, negative_output)
                val_loss += loss.item()
                
                # For simplicity, we'll consider anchor-positive as positive class (1) 
                # and anchor-negative as negative class (0)
                pos_sim = torch.nn.functional.cosine_similarity(anchor_output, positive_output)
                neg_sim = torch.nn.functional.cosine_similarity(anchor_output, negative_output)
                
                preds = (pos_sim > neg_sim).float()
                labels = torch.ones_like(preds)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "val_f1_score": val_f1
        })
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_pretrained_vit_model.pth')
            wandb.save('best_pretrained_vit_model.pth')  # Save the model file to wandb
            print("Saved new best model")
            wandb.run.summary["best_val_loss"] = best_val_loss
        
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        
        # Log confusion matrix to wandb
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["Negative", "Positive"])
        })
        
        print("\n")
        
        if scheduler.get_last_lr()[0] == 0:
                break

    wandb.finish()



num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = InfoNCE(temperature=0.07)

train_vit_model(model, train_loader, val_loader, num_epochs, device, 
                criterion, batch_size, num_pairs_train, 
                project_name="CVC-2024-Vit-embed", 
                run_name="ViT-InfoNCE-0.07",
                note = """Trained on 3/4 ratio on InfoNCE loss""")