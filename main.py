import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm
import csv
import os
from PIL import Image
import os, random
from torch.utils.data import DataLoader, random_split
import copy
import pandas as pd
import matplotlib.pyplot as plt


class SpaghettiDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        """
            ./combinedData/
                - clean
                - spaghetti
        """
        super().__init__()
        self.transform = transform
        self.samples = []

        for label, folder in enumerate(["clean", "spaghetti"]):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, fname), label))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path)
        if image.mode in ("P", "PA"):
            image = image.convert("RGBA")
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class SpaghettiCNN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# normalization constants from ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def train():
    epochs = 20
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # transforms to add randomness to the training data
    train_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2),
    ])

    # transforms to standardize the data for evaluation without any randomness
    eval_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    full_ds = SpaghettiDataset(data_dir="./combinedData", transform=train_transform)

    test_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - test_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])

    test_ds.dataset = copy.copy(full_ds)
    test_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model = SpaghettiCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'Epoch':>5}  {'Train Acc':>9}  {'Test Acc':>8}")
    print("─" * 28)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        correct, n = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for image, label in pbar:
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct += (output.argmax(1) == label).sum().item()
            n += len(label)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/n:.2%}")
        train_acc = correct / n


        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                correct += (model(image).argmax(1) == label).sum().item(); n += len(label)
        test_acc = correct / n
        print(f"{epoch:>5}  {train_acc:>8.2%}  {test_acc:>7.2%}")
        history.append((epoch, train_acc, test_acc))

        if test_acc >= 0.99:
            break

        scheduler.step()


    # save model
    model_name = f"CNN_{test_acc:.2%}_test_{epoch}epochs"
    model_path = f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)

    # save history of accuracies
    with open(f"history_{model_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_acc", "test_acc"])
        for epoch, train_acc, test_acc in history:
            writer.writerow([epoch, train_acc, test_acc])

    return model_path



def manual_test(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpaghettiCNN()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    class_names = ["clean", "spaghetti"]

    with torch.no_grad():
        for folder in class_names:
            correct = 0
            folder_path = os.path.join("./manual_tests", folder)
            print(f"Testing {folder}...")
            for path in os.listdir(folder_path):
                image = Image.open(os.path.join(folder_path, path))
                if image.mode in ("P", "PA"):
                    image = image.convert("RGBA")
                image = image.convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze()
                pred = logits.argmax(1).item()
                if class_names[pred] != folder:
                    print(f"\033[91m{path}: {class_names[pred]} (confidence: {probs[pred]:.2%})\033[0m")
                else:
                    print(f"{path}: {class_names[pred]} (confidence: {probs[pred]:.2%})")
                    correct += 1
            print(f"Correct: {correct}/{len(os.listdir(folder_path))} ({correct/len(os.listdir(folder_path)):.2%})")
            print("-" * 20)

def plot_accuracy():
    df = pd.read_csv("model_history.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df['epoch'], df['train_acc'] * 100, marker='o', markersize=4, linewidth=2, label='Train acc', color='#0000FF')
    ax.plot(df['epoch'], df['test_acc'] * 100, marker='s', markersize=4, linewidth=2, linestyle='--', label='Test acc', color='#FFA500')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 101)
    ax.set_xticks(df['epoch'])
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig("accuracy.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    model_path = train()
    print(model_path)
    manual_test(
        model_path
    )

    # plot_accuracy()
