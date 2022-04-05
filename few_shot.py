import clip
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from clip.clip import _convert_image_to_rgb, BICUBIC
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

from visualize import visualize


def get_features(dataloader):
    with torch.no_grad():
        features_all = []
        labels_all = []

        for images, labels in tqdm(dataloader):
            features = model.encode_image(images.to(device))

            features_all.append(features)
            labels_all.append(labels)

        features_all = torch.cat(features_all).cpu().numpy()
        labels_all = torch.cat(labels_all).cpu().numpy()

    return features_all, labels_all


if __name__ == "__main__":
    print(clip.available_models())

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device)

    preprocess = Compose([
            Resize((224, 224), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    # Prepare train data
    train_data = ImageFolder(root="./dataset/challenge/train", transform=preprocess)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    # Prepare test data
    test_data = ImageFolder(root="./dataset/challenge/validation", transform=preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    train_features, train_labels = get_features(train_loader)
    test_features, test_labels = get_features(test_loader)

    logistic_regressor = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    logistic_regressor.fit(train_features, train_labels)

    prediction = logistic_regressor.predict_proba(test_features)
    accuracy = np.mean(prediction.argmax(axis=-1) == test_labels) * 100

    print(f"accuracy: {accuracy}%")

    visualize(prediction, 10, "few_shot", f"{len(train_data) // 5}shot")

    with open(f"./result/few_shot/{len(train_data) // 5}shot/accuracy.txt", "w") as f:
        f.write(f"{accuracy}%")
