import argparse

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torchvision.transforms as transforms


def main():
    args = parse_arguments()
    if args.cpu or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
        # print('CUDA Device:', torch.cuda.get_device_name(device))
    model, _ = load_model_from_checkpoint(device=device)
    model.eval()
    transform = get_transform()
    with Image.open(args.image_path, 'r').convert('RGB') as image:
        if transform:
            image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        label = 'Cat' if predicted.item() else 'Dog'
        print(label)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cat-Dog Image Classification Inference')
    parser.add_argument('image_path', type=str)
    parser.add_argument('--cpu', action=argparse.BooleanOptionalAction, help='Whether to only use CPU')
    return parser.parse_args()


BEST_CAT_DOG_MODEL_A = \
    './checkpoint/ckpt-a-20250314-092618-383242.pth'


class CatDogModelA(nn.Module):

    def __init__(self):
        super(CatDogModelA, self).__init__()
        self.net = torchvision.models.efficientnet_b0(weights=None)
        for i in range(6):
            for param in self.net.features[i].parameters():
                param.requires_grad = False
        self.net.classifier[1] = nn.Linear(
            self.net.classifier[1].in_features, 2
        )
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x


def load_model_from_checkpoint(checkpoint_path=BEST_CAT_DOG_MODEL_A, *, device):
    model = CatDogModelA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    load_model(model, optimizer, checkpoint_path)
    return model, optimizer


def load_model(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint['model_state_dict']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    del checkpoint['optimizer_state_dict']
    # print('Model loaded:', checkpoint_path)
    return checkpoint


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    

if __name__ == '__main__':
    main()
