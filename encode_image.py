import io

import requests
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

import jax
from jax import pmap

from dalle_mini.dataset import *

from vqgan_jax.modeling_flax_vqgan import VQModel

model = VQModel.from_pretrained("flax-community/vqgan_f16_16384")
cc12m_images = '/media/storage/images'
cc12m_list = 'train-clean.tsv'
# cc12m_list = '/data/CC12M/images-10000.tsv'
cc12m_output = 'train-encoded.tsv'

image_size = 256
def image_transform(image):
    s = min(image.size)
    r = image_size / s
    s = (round(r * image.size[1]), round(r * image.size[0]))
    image = TF.resize(image, s, interpolation=InterpolationMode.LANCZOS)
    image = TF.center_crop(image, output_size = 2 * [image_size])
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    image = image.permute(0, 2, 3, 1).numpy()
    return image

dataset = CaptionDataset(
    images_root=cc12m_images,
    captions_path=cc12m_list,
    image_transform=image_transform,
    image_transform_type='torchvision',
    include_captions=False
)

def encode(model, batch):
#     print("jitting encode function")
    _, indices = model.encode(batch)
    return indices

def superbatch_generator(dataloader, num_tpus):
    iter_loader = iter(dataloader)
    for batch in iter_loader:
        superbatch = [batch.squeeze(1)]
        try:
            for b in range(num_tpus-1):
                batch = next(iter_loader)
                if batch is None:
                    break
                # Skip incomplete last batch
                if batch.shape[0] == dataloader.batch_size:
                    superbatch.append(batch.squeeze(1))
        except StopIteration:
            pass
        superbatch = torch.stack(superbatch, axis=0)
        yield superbatch

import os

def encode_captioned_dataset(dataset, output_tsv, batch_size=32, num_workers=16):
    if os.path.isfile(output_tsv):
        print(f"Destination file {output_tsv} already exists, please move away.")
        return
    
    num_tpus = 8    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    superbatches = superbatch_generator(dataloader, num_tpus=num_tpus)
    
    p_encoder = pmap(lambda batch: encode(model, batch))

    # We save each superbatch to avoid reallocation of buffers as we process them.
    # We keep the file open to prevent excessive file seeks.
    with open(output_tsv, "w") as file:
        iterations = len(dataset) // (batch_size * num_tpus)
        for n in tqdm(range(iterations)):
            superbatch = next(superbatches)
            encoded = p_encoder(superbatch.numpy())
            encoded = encoded.reshape(-1, encoded.shape[-1])

            # Extract fields from the dataset internal `captions` property, and save to disk
            start_index = n * batch_size * num_tpus
            end_index = (n+1) * batch_size * num_tpus
            paths = dataset.captions["image_file"][start_index:end_index].values
            captions = dataset.captions["caption"][start_index:end_index].values
            encoded_as_string = list(map(lambda item: np.array2string(item, separator=',', max_line_width=50000, formatter={'int':lambda x: str(x)}), encoded))
            batch_df = pd.DataFrame.from_dict({"image_file": paths, "caption": captions, "encoding": encoded_as_string})
            batch_df.to_csv(file, sep='\t', header=(n==0), index=None)
            
encode_captioned_dataset(dataset, cc12m_output, batch_size=64, num_workers=16)
