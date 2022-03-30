import clip
import torch
from torch.utils.data import DataLoader
from clip.clip import _convert_image_to_rgb
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, Normalize
from tqdm import tqdm

from visualize import visualize

if __name__ == "__main__":
    print(clip.available_models())

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device)

    preprocess = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    test_data = ImageFolder(root="./dataset/challenge", transform=preprocess)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    prompt = [
        "mannequin reflected by mirror",
        "person reflected by mirror",
        "mannequin",
        "person",
        "printed person image"
    ]
    text_inputs = torch.cat([clip.tokenize(c) for c in prompt]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    similarity_all = []
    predicted_all = []
    corrected = 0

    for x_batch, y_batch in tqdm(test_loader):
        image_inputs = x_batch.to(device)
        target = y_batch.to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity_all.append(similarity)

        predicted = similarity.argmax(dim=-1)
        predicted_all.append(predicted)

        corrected += predicted.eq(target).sum().item()

    similarity_all = torch.cat(similarity_all)
    predicted_all = torch.cat(predicted_all)

    visualize(similarity_all, 10, 0)
