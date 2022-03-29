import clip
import torch
import matplotlib.pyplot as plt
from clip.clip import _convert_image_to_rgb
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, Normalize

print(clip.available_models())

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load('RN50', device)

preprocess = Compose([
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

test_data = ImageFolder(root="./dataset/challenge", transform=preprocess)

plt.imshow(test_data[20][0].permute(1, 2, 0).numpy())
plt.show()

print(test_data[20][1])

# sampled = random.randint(0, len(cifar100) - 1)
#
# # Prepare the inputs
# image, class_id = cifar100[sampled]
# image = Image.open("./dataset/challenge/mannequin/2.png")
# image.show()
#
# image_input = preprocess(image).unsqueeze(0).to(device)
#
# text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in ["person not a mannequin", "mannequin"]]).to(device)
#
# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)
#
# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# print(similarity[0])
