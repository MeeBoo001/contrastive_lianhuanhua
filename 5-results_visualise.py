import matplotlib.pyplot as plt
from PIL import Image
import random
import json
import pandas as pd


# set the seed for reproducibility
random_seeds = [0, 1, 2, 3, 4]
for random_seed in random_seeds:
    random.seed(random_seed)

    mapping_table = pd.read_csv("annotations.csv")
    result_path = 'clusters.json'
    # image_dir = 'data/preprocessed_images'
    image_dir = "data/"

    # Load the clustering results
    with open(result_path, 'r') as f:
        clusters_dict = json.load(f)

    samples_per_cluster = 10

    print(clusters_dict.keys())

    n_clusters = len(clusters_dict)

    # Initialize the plot
    fig, axs = plt.subplots(
        n_clusters, samples_per_cluster,
        figsize=(samples_per_cluster*5, n_clusters*5))
    fig.suptitle('Sample Images from Each Cluster')

    # Iterate over each cluster
    for cluster_label, indices in clusters_dict.items():
        # Randomly sample images from this cluster
        sampled_indices = random.sample(indices, samples_per_cluster)

        # Plot each sampled image
        for ax, idx in zip(axs[int(cluster_label)], sampled_indices):
            # Load and display the image
            filename = mapping_table.filename[idx]
            img_path = f'{image_dir}/{filename}'
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        # Add a title to the first subplot in each row with the cluster label
        axs[int(cluster_label), 0].set_ylabel(f'Cluster {cluster_label}', size='large')

    # Save the visualization
    fig.savefig(f'cluster_samples{random_seed}.png')
