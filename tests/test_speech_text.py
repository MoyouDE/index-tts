from pathlib import Path
from types import SimpleNamespace

import pytest
from indextts.utils.speech_text import sanitize_speech_text


@pytest.mark.parametrize(('raw', 'expected'), [
    ('我要死掉惹❤\\~', '我要死掉惹.'),
    ('我要死掉惹❤️～～', '我要死掉惹.'),
    ('你好😊', '你好.'),
    ('你好👩🏽‍💻，欢迎！', '你好，欢迎！'),
    ('你好🇨🇳！', '你好！'),
    ('你好❤朋友', '你好朋友'),
    ('你好❤❤️朋友', '你好朋友'),
    ('你 好', '你 好'),
    ('Hello❤️world', 'Hello world'),
    ('你好~朋友', '你好,朋友'),
    ('“你好～～”', '“你好.”'),
    ('你好！~~~', '你好！'),
    ('真的？？？！！！', '真的?'),
    ('快跑！！！', '快跑!'),
    ('等等......', '等等…'),
    ('第一步1️⃣，第二步2️⃣。', '第一步1，第二步2。'),
    ('3~5天', '3-5天'),
    ('3 ～ 5天', '3 - 5天'),
    ('３～５天', '３-５天'),
    ('3.5~5.5米', '3.5-5.5米'),
    ('温度-5℃，折扣2.5%，价格￥30。', '温度-5℃，折扣2.5%，价格￥30。'),
    ('1+2=3，2×3=6', '1+2=3，2×3=6'),
    ("GPT-5, don't, state-of-the-art", "GPT-5, don't, state-of-the-art"),
    ('<行|xing2>，<银行|yin2 hang2>', '<行|xing2>，<银行|yin2 hang2>'),
    ('私はコーヒーが好きです。', '私はコーヒーが好きです。'),
    ('你好\u200b朋友\ufeff', '你好朋友'),
    ('你好\ufe0f', '你好'),
    ('直径⌀5', '直径⌀5'),
    ('❤\\~~~😊！！！', ''),
    ('……', ''),
    ('', ''),
])
def test_cleanup_preserves_words_and_normalizes_decorations(raw, expected):
    assert sanitize_speech_text(raw) == expected
    assert sanitize_speech_text(expected) == expected


@pytest.fixture(scope='module')
def normalizer():
    from indextts.utils.front import TextNormalizer
    result = TextNormalizer()
    result.load()
    return result


def test_number_and_pronunciation_normalization(normalizer):
    for raw, expected in [
        ('3~5天', '三到五天'),
        ('温度-5℃', '温度负五摄氏度'),
        ('<行|xing2>', '<行|xing2>'),
    ]:
        assert normalizer.normalize(raw) == expected


def test_web_engine_and_reader_use_identical_clean_tokens(normalizer):
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.runtime.engine import ReaderRuntime
    from indextts.runtime.tokenizer import ReaderTokenizer

    model_dir = Path(__file__).resolve().parents[1] / 'checkpoints'
    if not (model_dir / 'multilingual_zh_ja_yue_char_del.tiktoken').exists():
        pytest.skip('Local vocabulary needed for token parity check')
    tokenizer = ReaderTokenizer(model_dir)
    engine = IndexTTS2.__new__(IndexTTS2)
    engine.text_process = normalizer
    engine.tokenizer = tokenizer
    reader = ReaderRuntime.__new__(ReaderRuntime)
    reader.text_process = normalizer
    reader.tokenizer = tokenizer
    reader.device = 'cpu'
    reader.gpt = SimpleNamespace(text_pos_embedding=SimpleNamespace(emb=SimpleNamespace(num_embeddings=512)))

    reference = engine.prepare_text('我要死掉惹。', 'ZH')
    reference_tokens = tokenizer.encode('<|zh|> ' + reference, allowed_special='all') + [1]
    for text in ['我要死掉惹❤\\~', '我要死掉惹❤️～～', '我要死掉惹。']:
        prepared = engine.prepare_text(text, 'ZH')
        assert prepared == reference == '我要死掉惹.'
        assert reader._prepare_segments(text)[0].tolist() == [reference_tokens]
        assert tokenizer.encode('<|zh|> ' + prepared, allowed_special='all') + [1] == reference_tokens
    assert engine.prepare_text('你好❤\\~', 'ZH', text_normalization=False) == '你好.'

    with pytest.raises(ValueError, match='speakable'):
        engine.infer(None, '❤~~~', None, 'ZH')
    with pytest.raises(ValueError, match='speakable'):
        next(engine.infer_generator(None, '❤~~~', None, 'ZH'))
    with pytest.raises(ValueError, match='可朗读'):
        reader._prepare_segments('❤~~~')
