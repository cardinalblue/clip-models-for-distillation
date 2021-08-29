import torch
import os
import numpy as np

from model_components import CLIP, VisualTransformer
from preprocess import text_to_tensor, image_to_tensor, imagename_to_image

CWD = os.path.dirname(__file__)
trained_weight_file = os.path.join(CWD, 'weight', 'clip_model_cpu.pt')
assert os.path.exists(trained_weight_file)

def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval().float()

complete_clip_model = build_model(torch.load(trained_weight_file, map_location='cpu'))
papa_clip_visual = complete_clip_model.visual
baby_clip_visual = VisualTransformer(input_resolution=224,patch_size=32,width=384,layers=6,heads=384//16,output_dim=512)

papa_clip_visual.eval()
baby_clip_visual.eval()


class Clip:
    def __init__(self, model):
        self.model = model

    def image_to_vector(self, image, unit=False):
        with torch.no_grad():
            result = self.model.encode_image(image_to_tensor(image)).numpy()
        return result / np.linalg.norm(result) if unit else result

    def text_to_vector(self, text, unit=False):
        with torch.no_grad():
            result = self.model.encode_text(text_to_tensor(text)).numpy()
        return result / np.linalg.norm(result) if unit else result

clip = Clip(complete_clip_model)


if __name__ == '__main__':
    from PIL import Image
    print('Number of parameters in papa clip:', sum(p.numel() for p in papa_clip_visual.parameters()))
    print('Number of parameters in baby clip:', sum(p.numel() for p in baby_clip_visual.parameters()))
    print('===')
    image = Image.open(os.path.join(CWD, 'preprocess', 'cat.jpg'))
    image_vector = clip.image_to_vector(image, unit=True).squeeze()
    text_vector = clip.text_to_vector('cat', unit=True).squeeze()
    print('cosine similarity with word `cat`:', np.dot(image_vector, text_vector))
    text_vector = clip.text_to_vector('cat on bed', unit=True).squeeze()
    print('cosine similarity with word `cat on bed`:', np.dot(image_vector, text_vector))
