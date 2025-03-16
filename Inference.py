import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from Net.MultiLayerFusion import *
from skimage import transform as skt


class NoNan:
    def __call__(self, data):
        nan_mask = np.isnan(data)
        data[nan_mask] = 0.0
        data = np.expand_dims(data, axis=0)
        data /= np.max(data)
        return data

class Numpy2Torch:
    def __call__(self, data):
        data = torch.from_numpy(data)
        return data

class Resize:
    def __call__(self, data):
        data = skt.resize(data, output_shape=(96, 128, 96), order=1)
        return data

def load_model(model_path, device):

    model = IHFNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    img_numpy = nib.load(str(image_path)).get_fdata()
    img_tensor = transform(img_numpy)
    return img_tensor.float().unsqueeze(0)


def inference(mri_path, pet_path, cli_features, model, device, transform):

    mri_image = preprocess_image(mri_path, transform).to(device)
    pet_image = preprocess_image(pet_path, transform).to(device)
    cli_tensor = torch.from_numpy(np.array(cli_features)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(mri_image, pet_image, cli_tensor)
        prob = (outputs[0] + outputs[1] + outputs[2] + outputs[3]) / 4.0
        _, prediction = torch.max(prob, dim=1)

    return prediction.item(), prob.cpu().numpy()


if __name__ == "__main__":
    model_path = "path/to/model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    transform = transforms.Compose([
        Resize(),
        NoNan(),
        Numpy2Torch(),
        transforms.Normalize([0.5], [0.5])
    ])

    mri_path = "example/test_mri.nii"
    pet_path = "example/test_pet.nii"
    cli_features = [72.91, 1, 13, 205.1, 18.8, 5.1, 1, 0, 0]

    prediction, probability = inference(mri_path, pet_path, cli_features, model, device, transform)
    print(f"Predicted Class: {prediction}, Probability: {probability}")
