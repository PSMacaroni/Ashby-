"""
Loading CONCH or MUSK and using them for zero-shot tissue classification.

Typical use (same for both models):
    model, preprocess = load_conch()        # or: load_musk()
    text_emb = build_text_classifier(model)
    probs, img_emb = classify_images(model, preprocess, list_of_pil_images, text_emb)
"""
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import torch
from tqdm.auto import tqdm

from src.prompts import CLASSES, build_prompts

MUSK_REPO = "xiangjx/musk"


def load_conch(checkpoint_path=None, device=None, image_size=None):
    """
    Load CONCH. By default it downloads from Hugging Face (needs HF_TOKEN + approved access).
    If organizers preloaded the weights, pass checkpoint_path=".../pytorch_model.bin" instead.

    image_size: None = CONCH's native 448 px (images are resized to 448). 224 = run CONCH at
    224 px, which matches 224 px tiles (like NCT-CRC-HE-100K) without upscaling them.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    device = device or config.get_device()
    if checkpoint_path is None:
        if not config.HF_TOKEN:
            raise RuntimeError("HF_TOKEN is empty. Put your Hugging Face token in .env, "
                               "or pass checkpoint_path to preloaded weights.")
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
            device=device, hf_auth_token=config.HF_TOKEN, force_image_size=image_size)
    else:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", str(checkpoint_path), device=device, force_image_size=image_size)
    model.eval()  # inference mode (CONCH doesn't do this for us)
    return model, preprocess


class MuskTokenizer:
    """
    MUSK's text tokenizer: XLM-RoBERTa-style ids from MUSK's sentencepiece file, padded like
    musk.utils.xlm_tokenizer. Re-implemented because transformers 5 can no longer build
    XLMRobertaTokenizer from a .spm file (MUSK's README code fails with it).
    Matches transformers 4's XLMRobertaTokenizer: <s>=0, <pad>=1, </s>=2, <unk>=3, and every
    other sentencepiece id shifted by +1 (the "fairseq offset").
    """
    BOS, PAD, EOS, UNK = 0, 1, 2, 3

    def __init__(self, spm_path):
        import sentencepiece

        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(spm_path))

    def encode(self, text):
        # sentencepiece id 0 is its own <unk>, which maps to our UNK
        return [i + 1 if i != 0 else self.UNK for i in self.sp.encode(text)]

    def __call__(self, texts, max_len=100):
        """Returns (token_ids, padding_mask) tensors of shape [len(texts), max_len]."""
        ids_rows, pad_rows = [], []
        for text in texts:
            ids = [self.BOS] + self.encode(text)[:max_len - 2] + [self.EOS]
            n = len(ids)
            ids_rows.append(ids + [self.PAD] * (max_len - n))
            pad_rows.append([0] * n + [1] * (max_len - n))   # 1 = padding position
        return torch.tensor(ids_rows), torch.tensor(pad_rows)


class MuskModel:
    """Wraps MUSK so it has the same encode_image / encode_text / logit_scale as CONCH."""

    def __init__(self, model, tokenizer, dtype):
        self.model, self.tokenizer, self.dtype = model, tokenizer, dtype
        self.logit_scale = model.logit_scale

    def parameters(self):
        return self.model.parameters()

    def encode_image(self, images, **_):
        # with_head=True + out_norm=True is MUSK's setting for zero-shot (see its benchmarks/ code)
        emb = self.model(image=images.to(dtype=self.dtype), with_head=True, out_norm=True)[0]
        return emb.float()

    def encode_texts(self, texts):
        device = next(self.model.parameters()).device
        ids, pad = self.tokenizer(texts)
        emb = self.model(text_description=ids.to(device), padding_mask=pad.to(device),
                         with_head=True, out_norm=True)[1]
        return emb.float()


def load_musk(device=None):
    """
    Load MUSK (needs HF_TOKEN + approved access). Weights (1.35 GB) and tokenizer are cached in
    HF_HOME. MUSK's own loader would save them in ~/.cache instead, which is small on Delta.
    Uses float16 on GPU (as MUSK's README does) and float32 on CPU. Images are resized to 384 px.
    """
    import torchvision.transforms as T
    from huggingface_hub import hf_hub_download
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from timm.models import create_model
    from musk import modeling  # noqa: F401  (registers "musk_large_patch16_384" with timm)
    from musk import utils as musk_utils

    device = device or config.get_device()
    weights = hf_hub_download(MUSK_REPO, "model.safetensors", token=config.HF_TOKEN)
    spm_path = hf_hub_download(MUSK_REPO, "tokenizer.spm", token=config.HF_TOKEN)

    model = create_model("musk_large_patch16_384")
    musk_utils.load_model_and_may_interpolate(weights, model, "model|module", "")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model.to(device=device, dtype=dtype).eval()

    preprocess = T.Compose([
        T.Resize(384, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
    return MuskModel(model, MuskTokenizer(spm_path), dtype), preprocess


def tokenize(tokenizer, texts):
    """
    Same as conch.open_clip_custom.tokenize, but works with transformers 5.x too
    (CONCH's version calls batch_encode_plus, which transformers 5 removed).
    """
    tokens = tokenizer(texts, max_length=127, add_special_tokens=True,
                       return_token_type_ids=False, truncation=True,
                       padding="max_length", return_tensors="pt")
    # CONCH's context length is 128; the last slot is reserved, filled with <pad> for now
    return torch.nn.functional.pad(tokens["input_ids"], (0, 1), value=tokenizer.pad_token_id)


@torch.inference_mode()
def build_text_classifier(model, prompts=None):
    """
    Turn each class's prompts into one averaged text embedding.
    Returns a tensor of shape [n_classes, dim], rows in CLASSES order (dim: CONCH 512, MUSK 1024).
    """
    prompts = prompts or build_prompts()
    device = next(model.parameters()).device
    if isinstance(model, MuskModel):
        encode = model.encode_texts
    else:
        from conch.open_clip_custom import get_tokenizer

        tokenizer = get_tokenizer()
        encode = lambda texts: model.encode_text(tokenize(tokenizer, texts).to(device))
    rows = []
    for c in CLASSES:
        emb = encode(prompts[c])                  # [n_prompts, dim], already normalized
        emb = emb.mean(dim=0)                     # average the phrasings
        rows.append(emb / emb.norm())             # re-normalize to length 1
    return torch.stack(rows)


@torch.inference_mode()
def classify_images(model, preprocess, images, text_emb, batch_size=32):
    """
    Zero-shot classify a list of PIL images.
    Returns (probs, image_embeddings) as numpy arrays:
      probs            [n_images, n_classes], each row sums to 1 (columns in CLASSES order)
      image_embeddings [n_images, dim], keep these to train a small classifier later
    """
    device = text_emb.device
    scale = model.logit_scale.exp()
    all_probs, all_emb = [], []
    for i in tqdm(range(0, len(images), batch_size), desc="classifying", leave=False):
        batch = torch.stack([preprocess(img.convert("RGB")) for img in images[i:i + batch_size]])
        emb = model.encode_image(batch.to(device), proj_contrast=True, normalize=True)
        probs = (scale * emb @ text_emb.T).softmax(dim=-1)
        all_probs.append(probs.float().cpu().numpy())
        all_emb.append(emb.float().cpu().numpy())
    if not all_probs:
        return np.zeros((0, len(CLASSES))), np.zeros((0, text_emb.shape[1]))
    return np.concatenate(all_probs), np.concatenate(all_emb)
