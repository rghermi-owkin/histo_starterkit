import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Visualization and interpretability

def display_random_thumbnails(slide_paths, n_cols=5, n_rows=2):
    figsize = (n_cols * 4, n_rows * 4)
    plt.figure(figsize=figsize)

    random_indices = np.random.choice(
        len(slide_paths), 
        n_cols*n_rows, 
        replace=False,
    )
    for i, idx in enumerate(random_indices):
        slide = load_slide(slide_paths)
        
        thumbnail = slide.get_thumbnail(size=(200, 200))

        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.imshow(thumbnail)
        ax.set_title("")
        ax.set_axis_off()

    plt.show()

def display_slide(slide, mask=None, size=(600, 600)):
    thumbnail = slide.get_thumbnail(size=size)
    
    plt.figure(figsize=(12, 12))
    if mask is not None:
        resized_mask = (mask * 255).astype('uint8')
        resized_mask = Image.fromarray(resized_mask)
        resized_mask = resized_mask.resize(
            (thumbnail.size[0], thumbnail.size[1]), 
            Image.NEAREST,
        )
        resized_mask = np.expand_dims(
            np.array(resized_mask) / 255 > 0.6, 
            axis=2,
        )
        masked_thumbnail = np.array(thumbnail) * resized_mask
        plt.imshow(masked_thumbnail)
    else:
        plt.imshow(thumbnail)
    plt.axis('off')
    plt.show()

def display_embeddings(embeddings, clusters=None):
    # embeddings: numpy.array (N, 2)
    # clusters: numpy.array (N,)
    xs, ys = embeddings[:, 0], embeddings[:, 1]

    plt.figure(figsize=(10, 10))
    if clusters is not None:
        plt.scatter(xs, ys, s=10, c=clusters)
    else:
        plt.scatter(xs, ys, s=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()

def plot_logs_tts(train_losses, valid_losses, train_metrics, valid_metrics):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(train_losses, label='train_loss')
    plt.plot(valid_losses, label='valid_loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(train_metrics, label='train_metric')
    plt.plot(valid_metrics, label='valid_metric')
    plt.legend()
    
    plt.show()

def plot_logs_cv(
        cv_train_loss,
        cv_val_loss,
        cv_train_metric,
        cv_val_metric,
    ):
    cv_df = pd.DataFrame(data={ 
        'loss': cv_train_loss+cv_val_loss, 
        'metric': cv_train_metric+cv_val_metric,
        'set':['train']*len(cv_train_loss)+['valid']*len(cv_val_loss),
    })

    plt.figure(figsize=(12, 5))
    plt.subplot(121); sns.boxplot(x='loss', y='set', data=cv_df)
    plt.subplot(122); sns.boxplot(x='metric', y='set', data=cv_df)
    plt.show()