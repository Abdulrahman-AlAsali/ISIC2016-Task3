import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import timm

# Configuration
IMAGE_DIR = ".\ISBI2016_ISIC_Part3_Training_Data"
CSV_PATH = ".\ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
MODELS = ['resnet50', 'xception', 'efficientnet_b3']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
BATCH_SIZE = 32
FOLDS = 5
LR = 1e-4

# Dataset
class ISICDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        label = self.df.iloc[idx]["label"]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Model getter
def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == 'xception':
        model = timm.create_model('xception', pretrained=True, num_classes=1)
    elif model_name == 'efficientnet_b3':
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1)
    return model

# Training and validation
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# def validate(model, loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(DEVICE)
#             outputs = torch.sigmoid(model(images)).cpu().numpy()
#             all_preds.extend(outputs)
#             all_labels.extend(labels.numpy())
#     return roc_auc_score(all_labels, all_preds), all_preds, all_labels

def validate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(labels.numpy())

    preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)

    return auc, acc, f1, all_preds, all_labels

# Cross-validation training
def run_training(model_name, df, image_dir):
    skf = StratifiedKFold(n_splits=FOLDS)
    oof_preds = np.zeros(len(df))
    oof_labels = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n--- Fold {fold+1} ---")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = ISICDataset(train_df, image_dir, transform=train_transform)
        val_ds = ISICDataset(val_df, image_dir, transform=val_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = get_model(model_name).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_auc, val_acc, val_f1, preds, labels = validate(model, val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")


        oof_preds[val_idx] = np.array(preds).flatten()
        oof_labels[val_idx] = np.array(labels).flatten()

    auc = roc_auc_score(oof_labels, oof_preds)
    print(f"\nOverall AUC for {model_name}: {auc:.4f}")
    return oof_preds, oof_labels

# Main runner
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    all_preds = []
    for model_name in MODELS:
        preds, labels = run_training(model_name, df, IMAGE_DIR)
        all_preds.append(preds)

    # Ensemble (simple average)
    final_preds = np.mean(all_preds, axis=0)
    # final_auc = roc_auc_score(labels, final_preds)
    # print(f"\nEnsemble AUC: {final_auc:.4f}")
    final_preds_binary = [1 if p >= 0.5 else 0 for p in final_preds]
    final_auc = roc_auc_score(labels, final_preds)
    final_acc = accuracy_score(labels, final_preds_binary)
    final_f1 = f1_score(labels, final_preds_binary) 

    print(f"\nEnsemble AUC: {final_auc:.4f} | Accuracy: {final_acc:.4f} | F1: {final_f1:.4f}")
