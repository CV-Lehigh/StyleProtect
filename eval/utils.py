import os
import requests
import tempfile
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import linalg

import torch
from torchvision import transforms

from model import inception_model as inception

def download(url, ckpt_dir=None):
    name = url[url.rfind('/') + 1:]
    if ckpt_dir is None:
        ckpt_dir = tempfile.gettempdir()
    ckpt_dir = os.path.join(ckpt_dir, 'art_fid')
    ckpt_file = os.path.join(ckpt_dir, name)
    if not os.path.exists(ckpt_file):
        print(f'Downloading: \"{url[:url.rfind("?")]}\" to {ckpt_file}')
        if not os.path.exists(ckpt_dir): 
            os.makedirs(ckpt_dir)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        # first create temp file, in case the download fails
        ckpt_file_temp = os.path.join(ckpt_dir, name + '.temp')
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print('An error occured while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            # if download was successful, rename the temp file
            os.rename(ckpt_file_temp, ckpt_file)
    return ckpt_file

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        try:
            if isinstance(path, str):
                img = Image.open(path).convert('RGB')
            else:
                img = path
            if self.transforms is not None:
                img = self.transforms(img)
            return img
        except Exception as e:
            print(f'Error reading image {path}: {e}')
            
class ArtFID:
    def __init__(self):
        CKPT_URL = 'https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth'
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        ckpt_file = download(CKPT_URL)
        ckpt = torch.load(ckpt_file, map_location=self.device)

        model = inception.Inception3().to(self.device)
        model.load_state_dict(ckpt, strict=False)
        self.model = model.eval()
    def compute_fid(self, path_to_stylized, path_to_style):
        
        mu1, sigma1 = self.compute_activation_statistics(path_to_style)
        mu2, sigma2 = self.compute_activation_statistics(path_to_stylized)

        fid_value = self.compute_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value
    
    def get_activations(self, files, batch_size=50, num_workers=1):

        if batch_size > len(files):
            print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
            batch_size = len(files)
        dataset = ImagePathDataset(files, transforms=transforms.Compose([
                                    transforms.Resize((512, 512)),  
                                    transforms.ToTensor(),          
                                ]))
        
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers)

        pred_arr = np.empty((len(files), 2048))

        start_idx = 0

        pbar = tqdm(total=len(files))
        for batch in dataloader:
            batch = batch.to(self.device)

            with torch.no_grad():
                features = self.model(batch, return_features=True)

            features = features.cpu().numpy()
            pred_arr[start_idx:start_idx + features.shape[0]] = features
            start_idx = start_idx + features.shape[0]

            pbar.update(batch.shape[0])

        pbar.close()
        return pred_arr
    
    def compute_activation_statistics(self, files):

        act = self.get_activations(files)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def compute_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        def check_imaginary_component(value):
            if np.iscomplexobj(value):
                if not np.allclose(np.diagonal(value).imag, 0, atol=1e-3):
                    m = np.max(np.abs(value.imag))
                    raise ValueError(f'Imaginary component {m}')
                return value.real
        try:
            covmean = check_imaginary_component(covmean)
        except ValueError as e:
            print(e)

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    