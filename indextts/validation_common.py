"""Lightweight audio/context helpers and isolated session files."""
import re
import shutil
from pathlib import Path

def audio_details(path):
    import soundfile as sf
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("参考音频为空")
    return {"durationSeconds": info.duration, "sampleRate": info.samplerate,
            "channels": info.channels, "usedSeconds": min(info.duration, 15),
            "truncated": info.duration > 15}


def context_rows(rows):
    sentences = []
    for row in rows or []:
        if len(row) != 5:
            raise ValueError("每行需要句子 ID、章节、段落、类型、正文五列")
        sid, section, line, kind, text = row
        if not any(str(x or "").strip() for x in row):
            continue
        try:
            if isinstance(line, bool):
                raise ValueError()
            number = float(line)
            if not number.is_integer() or number < 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("段落编号必须为非负整数") from None
        kind = {"对白": "dialogue", "旁白": "narration"}.get(kind, kind)
        sentences.append({"sentenceId": str(sid or "").strip(), "sectionId": str(section or "").strip(),
                          "lineIndex": int(number), "sentenceType": kind, "text": str(text or "").strip()})
    return sentences


class SessionFiles:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()

    def session_dir(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("无效会话")
        path = (self.output_dir / session_id).resolve()
        if not path.is_relative_to(self.output_dir):
            raise ValueError("无效会话目录")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload(self, path, session_id, destination):
        source = Path(path).resolve()
        # Results from another server-side session must not be reused as inputs.
        if source.is_relative_to(self.output_dir) and not source.is_relative_to(self.session_dir(session_id)):
            raise ValueError("不能访问其他会话的文件")
        shutil.copyfile(source, destination)
        return destination
