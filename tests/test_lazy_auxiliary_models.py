from types import SimpleNamespace

import torch

import indextts.infer_v2_5 as inference


def test_qwen_constructor_does_not_load_weights(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Constructing unused QwenEmotion must not load a model")

    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", unexpected_load)
    monkeypatch.setattr(inference.AutoModelForCausalLM, "from_pretrained", unexpected_load)
    model = inference.QwenEmotion("not-a-model-directory", device="cpu")
    assert model.model is None and model.tokenizer is None


def test_reference_encoder_is_lazy_cached_and_preserves_features(monkeypatch):
    loads = []

    class Encoder(torch.nn.Module):
        def forward(self, input_features, **kwargs):
            return SimpleNamespace(hidden_states=[input_features * 2] * 18)

    def load(path, **kwargs):
        loads.append(path)
        return Encoder()

    monkeypatch.setattr(inference.Wav2Vec2BertModel, "from_pretrained", load)
    tts = inference.IndexTTS2.__new__(inference.IndexTTS2)
    tts.semantic_model = None
    tts.w2v_bert_dir = "reference-model"
    tts.device = "cpu"
    tts.semantic_mean = torch.tensor(1.0)
    tts.semantic_std = torch.tensor(2.0)
    features = torch.tensor([[[2.0, 3.0]]])
    for _ in range(2):
        output = tts.get_emb(features, torch.ones(1, 1))
        assert torch.equal(output, torch.tensor([[[1.5, 2.5]]]))
        assert not output.requires_grad
    assert loads == ["reference-model"]
