import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import lpips
from transformers import AutoProcessor, AutoModel
import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from piq import StyleLoss
from model.CSD import CSD_CLIP
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to avoid warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = lpips.LPIPS(net='alex').to(device)  

def convert_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict
    
csd = CSD_CLIP('vit_large', 'default')
checkpoint = torch.load('./model/checkpoint.pth', map_location="cpu", weights_only=False)  
state_dict = convert_state_dict(checkpoint['model_state_dict'])
csd.load_state_dict(state_dict, strict=False)
csd.eval()
csd.to(device)
csd_transform = transforms.Compose([
                transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
# Global model initialization
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

pick_processor = AutoProcessor.from_pretrained(processor_name_or_path)
pick_model = AutoModel.from_pretrained(model_pretrained_name_or_path)
pick_model.eval()  # Set to eval mode
pick_model = pick_model.to('cpu')  # Initially keep on CPU

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def calculate_ssim(img1, img2):
    """Calculating structural similarity index (SSIM) between two images."""
    # Convert PIL Image to numpy array
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    h, w = img1_np.shape[:2]
    win_size = min(h, w)
    if win_size % 2 == 0:
        win_size -= 1 
    return ssim(img1_np, img2_np, win_size=win_size, channel_axis=-1, full=True)[0]  # Return only the SSIM value, not the full diff map

def calculate_lpips(img1, img2):
    """Calculating Learned Perceptual Image Patch Similarity (LPIPS) between two images."""
    return loss_fn.forward(transform_image(img1).to(device), transform_image(img2).to(device)).item()


def calculate_CSD_cossim(imgs1, imgs2):
    """Calculating CSD cosine similarity between two images."""
    # Transform images
    img1 = [csd_transform(img).unsqueeze(0) for img in imgs1]
    img2 = [csd_transform(img).unsqueeze(0) for img in imgs2]
    
    # Extract features
    features1 = [extract_features(csd, img, use_cuda=True, use_fp16=True, eval_embed='head') for img in img1]
    features2 = [extract_features(csd, img, use_cuda=True, use_fp16=True, eval_embed='head') for img in img2]
    
    # Stack and average features
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    mean_feature1 = features1.mean(dim=0, keepdim=True)
    mean_feature2 = features2.mean(dim=0, keepdim=True)
    
    # Normalize mean features
    mean_feature1 = F.normalize(mean_feature1, p=2, dim=1)
    mean_feature2 = F.normalize(mean_feature2, p=2, dim=1)
    
    # Cosine similarity
    similarity = torch.mm(mean_feature1, mean_feature2.t())[0][0].item()
    return similarity


def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    image = transform(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return image


@torch.no_grad()
def extract_features(model, img, use_cuda=True, use_fp16=False, eval_embed='head'):
    """Extract features from a single image using the model."""
    # Ensure input is a batch
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    
    # Move to device
    if use_cuda:
        img = img.to(device)
    
    # Extract features
    if use_fp16:
        with torch.cuda.amp.autocast():
            bb_feats, cont_feats, style_feats = model(img)
            feats = style_feats if eval_embed == 'head' else bb_feats
    else:
        bb_feats, cont_feats, style_feats = model(img)
        feats = style_feats if eval_embed == 'head' else bb_feats
    
    return feats

style_loss_fn = StyleLoss()

def calculate_style_score(img1, img2):
    """
    Calculate style difference between two PIL images using Gram matrix-based style loss.
    Lower score means more similar in style.
    """
    # Preprocess PIL images
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),  
    ])
    img1_tensor = preprocess(img1).unsqueeze(0)  # shape: (1, 3, H, W)
    img2_tensor = preprocess(img2).unsqueeze(0)

    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    style_loss_fn.to(device)

    # Compute style score
    with torch.no_grad():
        score = style_loss_fn(img1_tensor, img2_tensor).item()
    return score

from utils import ArtFID
artfid = ArtFID()
def calculate_artfid(imgs1, imgs2):
    """Calculating ArtFID between two lists of images."""
    if len(imgs1) > len(imgs2):
        diff = len(imgs1) - len(imgs2)
        imgs2 += [imgs2[-1]] * diff
    elif len(imgs2) > len(imgs1):
        diff = len(imgs2) - len(imgs1)
        imgs1 += [imgs1[-1]] * diff

    assert len(imgs1) == len(imgs2)

    score = artfid.compute_fid(imgs1, imgs2)
    return score
