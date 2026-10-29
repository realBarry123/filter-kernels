import torch

def spectral_flatness(kernels, padding=0):
    spec = torch.fft.rfft(kernels, dim=-1).abs()
    power_spec = spec ** 2
    geo_mean = torch.exp(torch.log(power_spec).mean(dim=-1)) # numerically stable geometric mean
    ari_mean = power_spec.mean(dim=-1)
    return geo_mean / ari_mean