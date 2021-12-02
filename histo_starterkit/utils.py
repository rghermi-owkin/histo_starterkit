import os
import pickle
import PIL
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch

import openslide
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Category20

from histo_starterkit.constants import SLIDE_PATH
from histo_starterkit.models import MeanPool, MaxPool, Chowder, DeepMIL


# Load stuffs

def load_array(path):
    return np.load(path)

def load_pickle(path):
    return pickle.load(open(path, 'rb'))

def load_slide(slide_name, extension='tif'):
    slide_name = '.'.join([slide_name, extension])
    slide_path = os.path.join(SLIDE_PATH, slide_name)
    slide = openslide.open_slide(slide_path)
    return slide

def load_mask(slide_name, save_path):
    mask_name = '_'.join([slide_name, 'mask'])
    mask_name = '.'.join([mask_name, 'npy'])
    mask_path = os.path.join(save_path, 'masks', mask_name)
    
    mask = load_array(mask_path)
    return mask

def load_features(slide_name, save_path):
    features_name = '_'.join([slide_name, 'features'])
    features_name = '.'.join([features_name, 'npy'])
    features_path = os.path.join(save_path, 'features', features_name)

    features = load_array(features_path)
    return features

def load_metadata(slide_name, save_path):
    metadata_name = '_'.join([slide_name, 'metadata'])
    metadata_name = '.'.join([metadata_name, 'pkl'])
    metadata_path = os.path.join(save_path, 'metadata', metadata_name)

    metadata = load_pickle(metadata_path)
    return metadata

def get_model(model_name):
    if model_name == 'MeanPool':
        return MeanPool()
    if model_name == 'MaxPool':
        return MaxPool()
    if model_name == 'Chowder':
        return Chowder()
    if model_name == 'DeepMIL':
        return DeepMIL()


# Visualize stuffs

def display_random_thumbnails(df, n_cols=5, n_rows=2):
    figsize = (n_cols * 4, n_rows * 4)
    plt.figure(figsize=figsize)

    random_indices = np.random.choice(len(df), n_cols*n_rows, replace=False)
    for i, idx in enumerate(random_indices):
        slide = load_slide(df.iloc[idx].slide_name)
        
        thumbnail = slide.get_thumbnail(size=(200, 200))

        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.imshow(thumbnail)
        ax.set_title("")
        ax.set_axis_off()

    plt.show()

def display_thumbnail(slide, size=(400, 400)):
    thumbnail = slide.get_thumbnail(size=size)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(thumbnail)
    plt.axis('off')
    plt.show()
    
def display_mask(slide, mask, size=(500, 500)):
    thumbnail = slide.get_thumbnail(size=size)

    resized_mask = (mask * 255).astype("uint8")
    resized_mask = PIL.Image.fromarray(resized_mask)
    resized_mask = resized_mask.resize((thumbnail.size[0], thumbnail.size[1]), PIL.Image.NEAREST)
    
    masked_thumbnail = np.array(thumbnail) * np.expand_dims(np.array(resized_mask) / 255 > 0.6, axis=2)
    
    plt.figure(figsize=(12, 12))
    plt.subplot(221); plt.imshow(thumbnail); plt.axis('off')
    plt.subplot(222); plt.imshow(resized_mask, cmap='gray'); plt.axis('off')
    plt.subplot(212); plt.imshow(masked_thumbnail); plt.axis('off')
    plt.show()
    
def display_embeddings(embeddings, clusters=None):
    # embeddings: numpy.array (N, 2)
    xs, ys = embeddings[:, 0], embeddings[:, 1]

    plt.figure(figsize=(10, 10))
    if clusters is not None:
        plt.scatter(xs, ys, s=10, c=clusters)
    else:
        plt.scatter(xs, ys, s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()

def display_embeddings_interactive(embedding_df):
    # embedding_df: pd.DataFrame
    #     embedding_df.columns = ['x', 'y', 'cluster', 'image']
    
    output_notebook()

    datasource = ColumnDataSource(embedding_df)
    palette = Category20[embedding_df.cluster.nunique()]
    color_map = CategoricalColorMapper(factors=embedding_df.cluster.unique(),
                                       palette=palette)

    plot_figure = figure(
        title='UMAP projection of the tile features.',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Cluster:</span>
            <span style='font-size: 18px'>@cluster</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color={'field': 'cluster', 'transform': color_map},
        line_alpha=0.6,
        fill_alpha=0.6,
        size=10,
    )

    show(plot_figure)
    
    
# Training stuffs

def train_step(model, optimizer, criterion, train_dataloader, device):
    train_loss, all_preds, all_labels = [], [], []
    
    model.train()
    for batch in tqdm(train_dataloader):
        features, labels = batch
        features, labels = features.float().to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(features)
        preds = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach().cpu().numpy())
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
        
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    
    return train_loss, all_preds, all_labels

def valid_step(model, criterion, valid_dataloader, device):
    valid_loss, all_preds, all_labels = [], [], []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            features, labels = batch
            features, labels = features.float().to(device), labels.float().to(device)

            outputs, _ = model(features)
            preds = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)

            valid_loss.append(loss.detach().cpu().numpy())
            all_preds.append(preds.detach())
            all_labels.append(labels.detach())
            
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    
    return valid_loss, all_preds, all_labels

def fit(model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device):
    train_losses, valid_losses = [], []
    train_metrics, valid_metrics = [], []
    for epoch in range(num_epochs):

        train_loss, train_preds, train_labels = train_step(model, optimizer, criterion, train_dataloader, device)
        valid_loss, valid_preds, valid_labels = valid_step(model, criterion, valid_dataloader, device)
        
        train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
        train_metric = roc_auc_score(train_labels, train_preds)
        valid_metric = roc_auc_score(valid_labels, valid_preds)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_metrics.append(train_metric)
        valid_metrics.append(valid_metric)
        
        print('Epoch:', epoch+1)
        print('Training loss:', train_loss, '; Training metric:', train_metric)
        print('Validation loss:', valid_loss, '; Validation metric:', valid_metric)
        
    return train_losses, valid_losses, train_metrics, valid_metrics