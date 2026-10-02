"""Native full-source player and small Gradio browser callbacks, without model imports."""
from html import escape
from urllib.parse import quote


def player_html(path=None):
    if path is None:
        return '<div style="padding:12px;color:#777">上传素材后，在这里播放完整音轨并选择截取范围。</div>'
    url = "/gradio_api/file=" + quote(str(path).replace("\\", "/"), safe="/:")
    return (f'<audio id="producer-material-audio" aria-label="完整素材播放器" '
            f'controls preload="metadata" style="width:100%" src="{escape(url, quote=True)}"></audio>')


MARK_POSITION = """(value) => {
    const audio = document.getElementById('producer-material-audio');
    return audio && Number.isFinite(audio.currentTime) ? audio.currentTime : value;
}"""

PREVIEW_RANGE = """(start, end) => {
    const audio = document.getElementById('producer-material-audio');
    if (!audio || !Number.isFinite(audio.duration)) throw new Error('请先载入素材音轨');
    if (!(start >= 0 && start < end && end <= audio.duration + 0.01))
        throw new Error('截取起点必须小于终点，且在素材范围内');
    if (!audio.dataset.rangeBound) {
        audio.dataset.rangeBound = '1';
        audio.addEventListener('timeupdate', () => {
            const stop = Number(audio.dataset.rangeEnd);
            if (audio.dataset.rangeEnd && audio.currentTime >= stop) {
                audio.pause();
                audio.currentTime = stop;
            }
        });
        audio.addEventListener('pause', () => {
            delete audio.dataset.rangeEnd;
            cancelAnimationFrame(audio._rangeFrame);
        });
    }
    audio.dataset.rangeEnd = String(end);
    audio.currentTime = start;
    cancelAnimationFrame(audio._rangeFrame);
    const watch = () => {
        if (audio.paused || !audio.dataset.rangeEnd) return;
        const stop = Number(audio.dataset.rangeEnd);
        if (audio.currentTime >= stop) {
            audio.pause();
            audio.currentTime = stop;
        } else audio._rangeFrame = requestAnimationFrame(watch);
    };
    audio.play().then(() => { audio._rangeFrame = requestAnimationFrame(watch); })
        .catch(() => { delete audio.dataset.rangeEnd; });
    return [];
}"""
