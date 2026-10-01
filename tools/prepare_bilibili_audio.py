"""Fetch a public Bilibili audio track and optionally import it into the material workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(bvid, output_dir, workspace=None):
    if not re.fullmatch(r'BV[A-Za-z0-9]{10}', bvid):
        raise ValueError('请传入完整 BV 号')
    import requests
    url = f'https://www.bilibili.com/video/{bvid}/'
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': url})

    def api(endpoint, **params):
        response = session.get('https://api.bilibili.com/' + endpoint, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get('code') != 0:
            raise RuntimeError(f"Bilibili 接口失败：{result.get('code')} {result.get('message')}")
        return result['data']

    details = api('x/web-interface/view', bvid=bvid)
    pages = details['pages']
    if len(pages) != 1:
        raise ValueError('此脚本仅处理单 P 视频；请明确指定并单独准备需要的分 P')
    playback = api('x/player/playurl', bvid=bvid, cid=pages[0]['cid'], fnval=16, qn=80, fourk=0)
    tracks = playback.get('dash', {}).get('audio', [])
    if not tracks:
        raise RuntimeError('公开接口未提供独立音轨；未尝试登录或绕过访问限制')
    if playback.get('timelength', 0) / 1000 < details['duration'] - 2:
        raise RuntimeError('公开音轨仅包含预览，不能作为完整素材导入')
    track = max(tracks, key=lambda a: a['bandwidth'])
    if track.get('mimeType') != 'audio/mp4' or not track.get('codecs', '').startswith('mp4a.'):
        raise RuntimeError('当前音轨不是 AAC/MP4，不能按 m4a 保存')
    directory = Path(output_dir).expanduser().resolve() / bvid
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / (bvid + '.original.m4a')
    partial = destination.with_suffix('.m4a.part')
    try:
        response = session.get(track.get('baseUrl') or track['base_url'], stream=True, timeout=(30, 60))
        response.raise_for_status()
        with partial.open('xb') as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
        if not partial.stat().st_size:
            raise ValueError('下载音轨为空')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
        session.close()
    record = {'video': url, 'bvid': bvid, 'title': details['title'], 'cid': pages[0]['cid'],
        'durationSeconds': playback['timelength'] / 1000, 'audioBandwidth': track['bandwidth'],
        'audioCodec': track['codecs'], 'processing': 'Original public AAC retained; no denoising, trimming, or source separation.',
        'files': {destination.name: {'bytes': destination.stat().st_size, 'sha256': digest(destination)}}}
    (directory / 'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if workspace is not None:
        from indextts.material_service import MaterialService
        service = MaterialService(workspace)
        try:
            material = service.import_media(destination)
        finally:
            service.close()
        record['material'] = {'sourceId': material['sourceId'], 'audioSha256': material['audioSha256'],
            'durationSeconds': material['durationSeconds'], 'speechSeconds': material['speechSeconds'],
            'segmentCount': len(material['segments']), 'revision': material['revision']}
        (directory / 'source.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bvid')
    parser.add_argument('--output-dir', required=True, help='素材保存目录，不覆盖已有 BV 子目录')
    parser.add_argument('--workspace', help='可选：导入现有素材工作区，仅运行 FFmpeg 与 CPU VAD')
    args = parser.parse_args()
    print(json.dumps(prepare(args.bvid, args.output_dir, args.workspace), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
