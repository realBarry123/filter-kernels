import torch
from einops import rearrange

def generate_test_tones(freqs):

    def _generate_phase(freqs):
        """
        Given a (timesteps, voices)-tensor, output a (time, voices)-tensor
        of phases where time = timesteps * fs
        """
        delta = 2 * torch.pi * freqs / float(44100)
        initial_phase = torch.rand(freqs.shape[1]) * 2 * torch.pi
        phase = initial_phase.unsqueeze(0) + torch.cumsum(delta, axis=0)  # start from random phase
        phase = torch.fmod(phase, 2 * torch.pi) # wrap phase to prevent overflow
        return phase
    
    def _mix(audio):
        GAIN = 0.4
        """ 
        Sum the voice channels of a (channels, time, voices)-tensor, 
        normalize to int16 
        """
        mix = torch.sum(audio, dim=2)
        mix /= torch.max(torch.abs(mix))
        return (mix * 32767 * GAIN).to(torch.int16)
    
    phase = _generate_phase(freqs)
    audio = torch.sin(phase)
    stereo = audio.unsqueeze(0).repeat(2, 1, 1)
    stereo = _mix(stereo)

    return stereo
