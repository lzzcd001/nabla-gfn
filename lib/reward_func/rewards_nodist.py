import os
from PIL import Image
import io
import numpy as np
import time
import requests

import torch
from lib.utils import freeze

short_names = {
    "aesthetic_score": "aes",
    "imagereward": "imgr",
    "hpscore": "hps",
}
use_prompt = {
    "aesthetic_score": False,
    "imagereward": True,
    "hpscore": True,
}

def aesthetic_score(dtype=torch.float32, device="cuda", distributed=False):
    from lib.reward_func.aesthetic_scorer import AestheticScorer
    # why cuda() doesn't cause a bug?
    scorer = AestheticScorer(dtype=torch.float32, distributed=distributed).cuda() # ignore type;

    # input can be 3*256*256 or 3*1024*1024
    # @torch.no_grad() # original AestheticScorer already has no_grad()
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            # images = (images * 255).round().clamp(0, 255).to(torch.uint8)
            pass # assume float tensor in [0, 1]
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# For ImageReward
import ImageReward as RM
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def imagereward(dtype=torch.float32, device="cuda"):
    reward_model = RM.load("ImageReward-v1.0")
    reward_model.to(dtype).to(device)

    rm_preprocess = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _fn(images, prompts, metadata):
        dic = reward_model.blip.tokenizer(prompts,
                padding='max_length', truncation=True,  return_tensors="pt",
                max_length=reward_model.blip.tokenizer.model_max_length) # max_length=512
        device = images.device
        input_ids, attention_mask = dic.input_ids.to(device), dic.attention_mask.to(device)
        reward = reward_model.score_gard(input_ids, attention_mask, rm_preprocess(images)) # differentiable
        return reward.reshape(images.shape[0]).float(), {} # bf16 -> f32

    return _fn


# For HPSv2 reward
# https://github.com/tgxs002/HPSv2/blob/master/hpsv2/img_score.py
def hpscore(dtype=torch.float32, device=torch.device('cuda')):
    import huggingface_hub
    import torchvision.transforms.functional as F
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import root_path, hps_version_map
    from hpsv2.src.open_clip.transform import MaskAwareNormalize, ResizeMaxSize

    hps_version = "v2.1"
    model_dict = {}
    # def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            # device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()

    def _fn(images, prompts, metadata):

        image_size = model.visual.image_size[0]
        transforms = Compose([
            ResizeMaxSize(image_size, fill=0), # resize to 224x224
            MaskAwareNormalize(mean=model.visual.image_mean, std=model.visual.image_std),
        ])

        # these are not numerically identical, because
        # F.to_tensor(F.to_pil_image(img)) != img
        # due to RGB round up (in PIL it is 0~255 integer)

        # images = torch.stack([preprocess_val(F.to_pil_image(img)) for img in images.float()]).to(device)
        images = torch.stack([transforms(img) for img in images])
        texts = tokenizer(prompts).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = model(images, texts)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T # (bs, bs)
            hps_score = torch.diagonal(logits_per_image) # (bs,)

        return hps_score, {}

    return _fn

if __name__ == "__main__":
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer