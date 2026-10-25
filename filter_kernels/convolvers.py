import torch

def load_sao1_oobleck():
    from diffusers import AutoencoderOobleck
    return AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        subfolder="vae",
        torch_dtype=torch.float32
    )

def load_wav2vec2():
    from transformers import Wav2Vec2Model
    return Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        torch_dtype=torch.float32
    )

def load_hubert():
    from transformers import HubertModel
    return HubertModel.from_pretrained(
        "facebook/hubert-base-ls960",
        torch_dtype=torch.float32
    )


_MODELS = {
    "sao1_oobleck": load_sao1_oobleck,
    "wav2vec2": load_wav2vec2,
    "hubert": load_hubert
}

AVAILABLE_MODELS = list(_MODELS.keys())

def load(model_name):
    try: 
        return _MODELS[model_name]()
    except KeyError:
        raise NotImplementedError(f"loading \"{model_name}\" not implemented")