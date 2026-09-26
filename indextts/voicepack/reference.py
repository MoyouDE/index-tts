"""Reference-only conditioning producer; no speech generation networks."""

from pathlib import Path
import threading

import librosa
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.length_regulator import InterpolateRegulator
from indextts.utils.gpu_memory import staged_model, release_cuda_cache


class ReferenceProjection(nn.Module):
    """The checkpoint-compatible subset used by UnifiedVoice.get_emovec."""

    def __init__(self, config):
        super().__init__()
        emo = config.emo_condition_module
        self.spk_emb_proj = nn.Linear(192, config.model_dim)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024, output_size=emo.output_size, linear_units=emo.linear_units,
            attention_heads=emo.attention_heads, num_blocks=emo.num_blocks,
            input_layer=emo.input_layer)
        self.emo_perceiver_encoder = PerceiverResampler(
            1024, dim_context=emo.output_size, ff_mult=emo.perceiver_mult,
            heads=emo.attention_heads, num_latents=1)
        self.emo_layer = nn.Linear(config.model_dim, config.model_dim)
        self.emovec_layer = nn.Linear(1024, config.model_dim)

    def get_emovec(self, features, lengths):
        # Preserve the full path's transpose and length semantics exactly.
        encoded, mask = self.emo_conditioning_encoder(features, lengths)
        mask = self.emo_cond_mask_pad(mask.squeeze(1))
        value = self.emo_perceiver_encoder(encoded, mask).squeeze(1)
        return self.emo_layer(self.emovec_layer(value))


def load_reference_audio(path):
    """Keep the historical librosa mono/22050 preprocessing, then truncate."""
    audio, rate = librosa.load(path)
    if audio.size == 0 or not torch.isfinite(torch.from_numpy(audio)).all():
        raise ValueError("参考音频为空或包含非有限数值")
    return torch.tensor(audio).unsqueeze(0)[:, :15 * rate], rate


class ReferenceEncoder:
    def __init__(self, *, model_dir="checkpoints", cfg_path=None, device=None, use_bf16=False):
        self.model_dir = Path(model_dir).resolve()
        self.cfg = OmegaConf.load(cfg_path or self.model_dir / "config.yaml")
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("独立制包核心目前支持 CPU 或 CUDA")
        if use_bf16:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("BF16 制包需要支持 BF16 的 CUDA 设备")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("当前 CUDA 设备不支持 BF16")
        self.dtype = torch.bfloat16 if use_bf16 else None
        self.model_version = self.cfg.get("version", "2.5")
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        # Keep new modules local until all strict checkpoint checks succeed.
        projection = ReferenceProjection(self.cfg.gpt)
        state = torch.load(self.model_dir / self.cfg.gpt_checkpoint, map_location="cpu")
        state = state.get("model", state)
        prefixes = tuple(name + "." for name in (
            "spk_emb_proj", "emo_conditioning_encoder", "emo_perceiver_encoder", "emo_layer", "emovec_layer"))
        projection.load_state_dict({k: v for k, v in state.items() if k.startswith(prefixes)}, strict=True)
        del state
        projection.eval()
        if self.dtype is not None:
            projection.bfloat16()

        args = self.cfg.s2mel.length_regulator
        regulator = InterpolateRegulator(
            channels=args.channels, sampling_ratios=args.sampling_ratios,
            is_discrete=args.is_discrete, in_channels=args.get("in_channels"),
            vector_quantize=args.get("vector_quantize", False), codebook_size=args.content_codebook_size,
            n_codebooks=args.get("n_codebooks", 1), quantizer_dropout=args.get("quantizer_dropout", 0.0),
            f0_condition=args.get("f0_condition", False), n_f0_bins=args.get("n_f0_bins", 512))
        state = torch.load(self.model_dir / self.cfg.s2mel_checkpoint, map_location="cpu")
        regulator.load_state_dict({k.removeprefix("module."): v for k, v in
                                   state["net"]["length_regulator"].items()}, strict=True)
        del state
        regulator.eval()
        camp = CAMPPlus(feat_dim=80, embedding_size=192)
        camp.load_state_dict(torch.load(self.model_dir / "hf_cache/campplus_cn_common.bin", map_location="cpu"), strict=True)
        camp.eval()
        encoder_dir = self.model_dir / "hf_cache/w2v-bert-2.0"
        extractor = SeamlessM4TFeatureExtractor.from_pretrained(encoder_dir, local_files_only=True)
        semantic = Wav2Vec2BertModel.from_pretrained(encoder_dir, local_files_only=True).eval()
        stats = torch.load(self.model_dir / self.cfg.w2v_stat, map_location="cpu")
        emotion = torch.load(self.model_dir / self.cfg.emo_matrix, map_location="cpu")
        speaker = torch.load(self.model_dir / self.cfg.spk_matrix, map_location="cpu")
        self.projection, self.regulator, self.camp = projection, regulator, camp
        self.semantic_model, self.extract_features = semantic, extractor
        self.semantic_mean, self.semantic_std = stats["mean"], torch.sqrt(stats["var"])
        self.emotion_matrix, self.speaker_matrix = emotion, speaker
        self._loaded = True

    @torch.no_grad()
    def extract_voice_conditioning(self, audio_path, verbose=False):
        with self._lock:
            audio, rate = load_reference_audio(audio_path)
            self._load()
            audio_22k = audio if rate == 22050 else torchaudio.functional.resample(audio, rate, 22050)
            audio_16k = audio if rate == 16000 else torchaudio.functional.resample(audio, rate, 16000)
            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            with staged_model(self.semantic_model, self.device) as model:
                output = model(input_features=inputs["input_features"].to(self.device),
                               attention_mask=inputs["attention_mask"].to(self.device), output_hidden_states=True)
                features = (output.hidden_states[17] - self.semantic_mean.to(self.device)) / self.semantic_std.to(self.device)
                del output
            spect = self.cfg.s2mel.preprocess_params.spect_params
            ref_mel = mel_spectrogram(audio_22k.to(self.device).float(), n_fft=spect.n_fft,
                win_size=spect.win_length, hop_size=spect.hop_length, num_mels=spect.n_mels,
                sampling_rate=self.cfg.s2mel.preprocess_params.sr, fmin=spect.get("fmin", 0),
                fmax=None if spect.get("fmax", "None") == "None" else 8000, center=False)
            lengths = torch.tensor([ref_mel.size(2)], dtype=torch.long, device=self.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(self.device), num_mel_bins=80,
                                                   dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            with staged_model(self.camp, self.device) as model:
                style = model(feat.unsqueeze(0))
            with staged_model(self.regulator, self.device) as model:
                prompt = model(features, ylens=lengths, n_quantizers=3, f0=None)[0]
            cond_lengths = torch.tensor([features.shape[-1]], device=self.device)
            with staged_model(self.projection, self.device) as model:
                with torch.amp.autocast(self.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    base = model.get_emovec(features, cond_lengths)
                    speaker = model.spk_emb_proj(style)
            speaker_matrix = torch.split(self.speaker_matrix.to(self.device), list(self.cfg.emo_num))
            emotion_matrix = torch.split(self.emotion_matrix.to(self.device), list(self.cfg.emo_num))
            indexes = [torch.argmax(F.cosine_similarity(style.float(), matrix.float(), dim=1))
                       for matrix in speaker_matrix]
            basis = torch.cat([matrix[index].unsqueeze(0) for index, matrix in zip(indexes, emotion_matrix)], dim=0)
            return {name: value.detach().cpu().contiguous() for name, value in {
                "speaker_latent": speaker, "base_emotion": base, "emotion_basis": basis,
                "prompt_condition": prompt, "ref_mel": ref_mel.float(), "speaker_style": style.float(),
            }.items()}

    def close(self):
        with self._lock:
            for name in ("projection", "regulator", "camp", "semantic_model", "extract_features",
                         "semantic_mean", "semantic_std", "emotion_matrix", "speaker_matrix"):
                if hasattr(self, name):
                    delattr(self, name)
            self._loaded = False
            release_cuda_cache(self.device)
