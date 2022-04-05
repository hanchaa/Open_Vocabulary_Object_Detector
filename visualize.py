import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as img
from matplotlib import gridspec
from numpy import ndarray


def visualize(accuracy: ndarray, images_per_class, method, tag):
    labels = ["m_mannequin", "m_person", "mannequin", "person", "person_image"]
    num_labels = len(labels)

    if os.path.exists(f"./result/{method}/{tag}") is False:
        os.makedirs(f"./result/{method}/{tag}")

    for idx, label in enumerate(labels):
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(nrows=5, ncols=4, width_ratios=[2, 7, 2, 7])
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)

        for cnt in range(images_per_class):
            image = img.imread(f"./dataset/challenge/validation/{labels[idx]}/{cnt + 1}.jpg")
            ax0 = plt.subplot(gs[cnt * 2])
            ax0.imshow(image)

            ax1 = plt.subplot(gs[cnt * 2 + 1])
            ax1.bar(np.arange(num_labels), accuracy[idx * images_per_class + cnt])

            plt.sca(ax1)
            plt.xticks(np.arange(num_labels), labels)

        plt.show()
        fig.savefig(f"./result/{method}/{tag}/{label}.png")
