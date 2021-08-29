# clip-models-for-distillation

This repo contains the source code for original and distilled models for CLIP. It also contains a simple API to use the original CLIP model.

Most of the code is taken from [the original implementation](https://github.com/openai/CLIP) by OpenAI. It is just organized in a different manner to make it easy to understand.

### Related article
https://tech.pic-collage.com/distillation-of-clip-model-and-other-experiments-f8394b7321ce?gi=1d6a8777178a


## Things to do before using this repo

1. Download the trained weight for the original CLIP model from [this link](https://drive.google.com/file/d/1eiAhMWSSE30E-LgXi7DuAW2LgiLp80Q6/view?usp=sharing) and put it into the folder `weight`. 
2. Create a new environment and install dependencies using `pip install -r requirements.txt`.

## How to perform distillation

A rough outline is (refer to the article for more details):

```python
from models import papa_clip_visual, baby_clip_visual

optimizer = AdamW(baby_clip_visual.parameters())

for x in loader:
    y_s = baby_clip_visual(x)
    y_t = papa_clip_visual(x)
    loss = loss_function(y_s, y_t)
    optimizer.zero_grad()
    loss.backward()
    ...
```

## How to use the original CLIP model

```python
image = Image.open('path/to/image.jpg')

from models import clip
image_vector = clip.image_to_vector(image)
text_vector = clip.text_to_vector('my text')
```