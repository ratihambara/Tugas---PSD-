import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.getData import data_a
from torchvision import models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def main():
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6
    folds = [1, 2, 3, 4, 5]
    orig_path = "../Original Images/Original Images/FOLDS"
    for fold in folds:
        test_loader = DataLoader(data_a(ori=orig_path, folds=folds, subdir='Test'), batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[-1] =nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = torch.load('trained_modelmobilenet.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss_batch = checkpoint['loss']

    prediction, ground_truth = [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)
            prediction.extend(torch.argmax(pred,dim=1).detach().numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).detach().numpy())

    classes = ('Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox')

    # Build confusion matrix
    cf_matrix = confusion_matrix(ground_truth, prediction)
    print(cf_matrix)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')


    print("accuracy score = ", accuracy_score(ground_truth, prediction))
    print("precision score = ", precision_score(ground_truth, prediction, average='weighted'))
    print("recall score = ", recall_score(ground_truth, prediction, average='weighted'))
    print(f"f1 score score =  {f1_score(ground_truth, prediction, average='weighted')*100:.2f}%")


if __name__ == "__main__":
    main()
