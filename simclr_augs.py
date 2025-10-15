# simclr_augs.py
import torch
from torchvision import transforms as T

# keep your original order as a list of (name, factory) pairs
# add near your other imports
from PIL import Image
import torchvision.transforms as T

def _blur_kernel(image_size):
    # ~10% of min side, forced odd
    k = int(0.1 * image_size)
    return (k // 2) * 2 + 1

def aug_catalog(image_size=224):
    return [
        ("topil",     lambda: T.ToPILImage()),
        # --- new SimCLR-ish ops tuned for 224 ---
        ("rrc50",     lambda: T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=Image.BICUBIC)),
        ("rrc35",     lambda: T.RandomResizedCrop(image_size, scale=(0.35, 1.0), interpolation=Image.BICUBIC)),
        ("cj_s05",    lambda: T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)),  # strength≈0.5
        ("cj_s08",    lambda: T.RandomApply([T.ColorJitter(0.64,0.64,0.64,0.16)], p=0.8)),# strength≈0.8
        ("gray",      lambda: T.RandomGrayscale(p=0.2)),
        ("blur_p03",  lambda: T.RandomApply([T.GaussianBlur(kernel_size=_blur_kernel(image_size),
                                                            sigma=(0.1, 2.0))], p=0.3)),
        ("blur_p07",  lambda: T.RandomApply([T.GaussianBlur(kernel_size=_blur_kernel(image_size),
                                                            sigma=(0.1, 2.0))], p=0.7)),
        # --- your original ops ---
        ("rotate",    lambda: T.RandomRotation(20)),
        ("affine",    lambda: T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))),
        ("hflip",     lambda: T.RandomHorizontalFlip(p=0.5)),
        ("totensor",  lambda: T.ToTensor()),
        ("noise01",   lambda: T.Lambda(lambda x: x + 0.01 * torch.randn_like(x))),  # gentler noise for 300 imgs
        ("clamp",     lambda: T.Lambda(lambda x: torch.clamp(x, 0, 1))),
        ("normalize", lambda: T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])),
    ]

def build_transform(selected_names=None, image_size=224):
    catalog = aug_catalog(image_size=image_size)
    if not selected_names:
        ops = [factory() for _, factory in catalog]
    else:
        keep = {name.strip().lower() for name in selected_names}
        ops = [factory() for name, factory in catalog if name in keep]
    return T.Compose(ops)


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)
