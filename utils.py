import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from main import SpaghettiCNN

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["clean", "spaghetti"]
# MODEL_PATH = "100%manual_CNN_98.70%_test_20epochs.pth"
MODEL_PATH = "model.pth"

_model = {}

def _get_model():
    if "model" not in _model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SpaghettiCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        _model["model"] = model
        _model["device"] = device
    return _model["model"], _model["device"]


def _preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    if image.mode in ("P", "PA"):
        image = image.convert("RGBA")
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def classify_image(image: Image.Image):
    model, device = _get_model()
    tensor = _preprocess(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    probs_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    return CLASS_NAMES[pred_idx], confidence, probs_dict


def generate_gradcam(image: Image.Image) -> Image.Image:
    model, device = _get_model()
    tensor = _preprocess(image).to(device)

    target_layer = model.features[7]

    activations, gradients = [], []

    def fwd_hook(_module, _input, output):
        activations.append(output.detach())

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_h = target_layer.register_forward_hook(fwd_hook)
    bwd_h = target_layer.register_full_backward_hook(bwd_hook)

    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, pred_idx].backward()

    fwd_h.remove()
    bwd_h.remove()

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().cpu().numpy()

    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[:, :, :3].astype(np.float32)

    if image.mode in ("P", "PA"):
        image = image.convert("RGBA")
    img_rgb = image.convert("RGB").resize((224, 224))
    img_arr = np.array(img_rgb).astype(np.float32) / 255.0

    overlay = 0.5 * img_arr + 0.5 * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(overlay)
