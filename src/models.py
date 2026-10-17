"""
Loading CONCH and using it for zero-shot tissue classification.

Typical use:
    model, preprocess = load_conch()
    text_emb = build_text_classifier(model)
    probs, img_emb = classify_images(model, preprocess, list_of_pil_images, text_emb)
"""
from src import config  # first, so HF_HOME from .env is applied

import numpy as np
import torch
from tqdm.auto import tqdm

from src.prompts import CLASSES, build_prompts


def load_conch(checkpoint_path=None, device=None):
    """
    Load CONCH. By default it downloads from Hugging Face (needs HF_TOKEN + approved access).
    If organizers preloaded the weights, pass checkpoint_path=".../pytorch_model.bin" instead.
    """
    from conch.open_clip_custom import create_model_from_pretrained

    device = device or config.get_device()
    if checkpoint_path is None:
        if not config.HF_TOKEN:
            raise RuntimeError("HF_TOKEN is empty. Put your Hugging Face token in .env, "
                               "or pass checkpoint_path to preloaded weights.")
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
            device=device, hf_auth_token=config.HF_TOKEN)
    else:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", str(checkpoint_path), device=device)
    model.eval()  # inference mode (CONCH doesn't do this for us)
    return model, preprocess


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
    Returns a tensor of shape [n_classes, 512], rows in CLASSES order.
    """
    from conch.open_clip_custom import get_tokenizer

    prompts = prompts or build_prompts()
    tokenizer = get_tokenizer()
    device = next(model.parameters()).device
    rows = []
    for c in CLASSES:
        tokens = tokenize(tokenizer, prompts[c]).to(device)
        emb = model.encode_text(tokens)          # [n_prompts, 512], already normalized
        emb = emb.mean(dim=0)                     # average the phrasings
        rows.append(emb / emb.norm())             # re-normalize to length 1
    return torch.stack(rows)


@torch.inference_mode()
def classify_images(model, preprocess, images, text_emb, batch_size=32):
    """
    Zero-shot classify a list of PIL images.
    Returns (probs, image_embeddings) as numpy arrays:
      probs            [n_images, n_classes], each row sums to 1 (columns in CLASSES order)
      image_embeddings [n_images, 512], keep these to train a small classifier later
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
