import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

from metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_CSD_cossim, calculate_artfid

def get_args_parser():
    parser = argparse.ArgumentParser()
    #1. setup image paths
    parser.add_argument("--original_image_path", required=True, type=str, help="the path to the original images")
    parser.add_argument("--protected_image_path", required=True, type=str, help="the path to the protected images")
    parser.add_argument("--protected_image_gen_path", required=True, type=str, help="the path to the protected edits")
    #3. setup save path
    parser.add_argument("--save_path", required=True, type=str, help="the path to save the results")
    return parser

def main(args):

    original_image_path = args.original_image_path
    protected_image_path = args.protected_image_path
    protected_image_gen_path = args.protected_image_gen_path


    invisible_record = {'psnr': [], 'ssim': [], 'lpips': []} 
    style_record = {'csd': [], 'artfid': []}
    for artist in tqdm(os.listdir(protected_image_path)):
        prompt = artist.replace('-', ' ')
        print(f"processing {prompt}...")
        protected_images_path = sorted([os.path.join(protected_image_path, artist, image) for image in os.listdir(os.path.join(protected_image_path, artist))])
        original_images_path = sorted([os.path.join(original_image_path, artist, image) for image in os.listdir(os.path.join(protected_image_path, artist))])
        protected_image_gens_path = sorted([os.path.join(protected_image_gen_path, artist, image) for image in os.listdir(os.path.join(protected_image_gen_path, artist))])

        original_images = [Image.open(image_path).resize((512, 512)) for image_path in original_images_path]
        protected_images = [Image.open(image_path).resize((512, 512)) for image_path in protected_images_path]
        protected_image_gens = [Image.open(image_path) for image_path in protected_image_gens_path]

        # 1. check invisible perturbation
        for original_image, protected_image in zip(original_images, protected_images):
            
            invisible_record['psnr'].append(float(calculate_psnr(original_image, protected_image)))
            invisible_record['ssim'].append(float(calculate_ssim(original_image, protected_image)))
            invisible_record['lpips'].append(float(calculate_lpips(original_image, protected_image)))

        # 2. check style replication rate
        style_record['csd'].append(float(calculate_CSD_cossim(original_images, protected_image_gens)))
        style_record['artfid'].append(float(calculate_artfid(original_images, protected_image_gens)))


    invisible_record['psnr'] = np.mean(invisible_record['psnr'])
    invisible_record['ssim'] = np.mean(invisible_record['ssim'])
    invisible_record['lpips'] = np.mean(invisible_record['lpips'])

    style_record['csd'] = np.mean(style_record['csd'])
    style_record['artfid'] = np.mean(style_record['artfid'])

    
    print(f"evaluating {args.save_path.split('/')[-1].split('.')[0]}...")
    print(f"invisible perturbation: PSNR: {invisible_record['psnr']} , SSIM: {invisible_record['ssim']} , LPIPS: {invisible_record['lpips']}")
    print(f"style replication rate: CSD: {style_record['csd']} , ArtFID: {style_record['artfid']}")

    # 4. save the results
    with open(args.save_path, 'w') as f:
        json.dump(invisible_record, f, indent=4)
        json.dump(style_record, f, indent=4)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)