import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from torchvision import models 
from utils.getData import data_a

def main():
    BATCH_SIZE = 32
    EPOCH = 25
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6  # Adjust with the number of classes
    folds = [1, 2, 3, 4, 5]

    # Paths to dataset
    orig_path = "../Original Images/Original Images/FOLDS"
    aug_path = "../Augmented Images/Augmented Images/FOLDS_AUG"

    # Initialize dataset
    for fold in folds:
        train_data = DataLoader(ConcatDataset([data_a(aug=aug_path, folds=fold, subdir='Train'),
                                               data_a(ori=orig_path, folds=fold, subdir='Train')]),
                                               batch_size=BATCH_SIZE, shuffle=True)
        val_data = DataLoader(data_a(ori=orig_path, folds=folds, subdir='Valid'), batch_size=BATCH_SIZE, shuffle=False)

    # Define model, loss function, and optimizer
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[-1] =nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(EPOCH):
        # Training phase
        model.train()
        loss_train, correct_train, total_train = 0.0, 0, 0

        for src, trg in train_data:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        train_losses.append(loss_train / len(train_data))

        # Validation phase
        model.eval()
        loss_val, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for src, trg in val_data:
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)

                loss_val += loss.item()
                _, predicted = torch.max(pred, 1)
                total_val += trg.size(0)
                correct_val += (predicted == trg).sum().item()

        accuracy_val = 100 * correct_val / total_val
        val_losses.append(loss_val / len(val_data))

        print(f"Epoch [{epoch + 1}/{EPOCH}], "
              f"Train Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy_train:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy_val:.2f}%")

    
    # Save the trained model
    torch.save(model.state_dict(), "trained_modelmobilenet.pt")

    # Plot training and validation loss
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Train Loss')
    plt.plot(range(EPOCH), val_losses, color="#ff5733", label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("./train_val_loss.png")
    plt.show()

if __name__ == "__main__":
    main()