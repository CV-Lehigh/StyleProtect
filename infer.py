import os
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--prompts", type=str, default=["A boy riding a bicycle on a sunny street",
        "A dog sitting under a tree in a park",
        "An old lady reading a newspaper on a bench"])
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.path
    prompts = args.prompts
    torch.manual_seed(9222)
    output_path = args.output_path
    for artist in tqdm(os.listdir(path), desc="artist loop"):
        if os.path.exists(f"{output_path}/{artist}"):
            continue
        pipeline = DiffusionPipeline.from_pretrained(f"{path}/{artist}", torch_dtype=torch.float16, use_safetensors=True, safety_checker=None).to("cuda:0")
        for prompt in tqdm(prompts, desc="prompt loop"):
            print(prompt+" in sks style")
            image = pipeline(prompt+" in sks style", num_inference_steps=50, guidance_scale=7.5).images[0]
            os.makedirs(f"{output_path}/{artist}", exist_ok=True)
            if os.path.exists(f"{output_path}/{artist}/{prompt.replace(' ', '_')}.png"):
                continue
            image.save(f"{output_path}/{artist}/{prompt.replace(' ', '_')}.png")