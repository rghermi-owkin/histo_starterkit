import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from sklearn.metrics import roc_auc_score

import mlflow


# Training and validation

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    device: str,
    ):

    model.train()

    train_loss = []
    all_preds, all_labels = [], []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Get data
        features, mask, labels = batch

        # Put on device
        features = features.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        # Compute outputs and loss
        outputs, _ = model(features, mask)
        preds = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)

        # Run backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save logs
        train_loss.append(loss.detach().cpu().numpy())
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    return train_loss, all_preds, all_labels

def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module, 
    device: str,
    ):

    model.eval()

    valid_loss = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Get data
            features, mask, labels = batch

            # Put on device
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Compute outputs and loss
            outputs, _ = model(features, mask)
            preds = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)

            # Save logs
            valid_loss.append(loss.detach().cpu().numpy())
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    return valid_loss, all_preds, all_labels

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    ):

    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Get data
            features, mask, labels = batch

            # Put on device
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Compute outputs and loss
            outputs, _ = model(features, mask)
            preds = torch.sigmoid(outputs)

            # Save logs
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    return all_preds, all_labels

def fit(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    verbose: bool = True,
    ):

    train_losses, valid_losses = [], []
    train_metrics, valid_metrics = [], []
    for epoch in range(num_epochs):
        # Training step
        tr_losses, tr_preds, tr_labels = train_step(
            model, train_dataloader, criterion, optimizer, device)
        # Validation step
        val_losses, val_preds, val_labels = eval_step(
            model, valid_dataloader, criterion, device)
        
        # Compute loss and metric
        train_loss = np.mean(tr_losses)
        valid_loss = np.mean(val_losses)
        try:
            train_metric = roc_auc_score(tr_labels, tr_preds)
        except ValueError:
            train_metric = -1
        try:
            valid_metric = roc_auc_score(val_labels, val_preds)
        except ValueError:
            valid_metric = -1
        
        # Save logs
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)

        logs = {
            'train_loss':train_loss,
            'valid_loss':valid_loss,
            'train_metric':train_metric,
            'valid_metric':valid_metric,
        }
        mlflow.log_metrics(logs, step=epoch)
        
        if verbose:
            print('Epoch:', epoch+1)
            print('Train loss:', train_loss, '; Valid loss:', valid_loss)
            print('Train metric:', train_metric, '; Valid metric:', valid_metric)
        
    return train_losses, valid_losses, train_metrics, valid_metrics