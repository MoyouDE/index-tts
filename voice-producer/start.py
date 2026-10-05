"""Start the reference-audio voice producer, without emotion or audition pages."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# This directory is a tool entrypoint; the implementation stays in indextts.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    from indextts.validation_web import main as web_main
    web_main([
        "--modules", "producer,audition", "--producer-workflow",
        "--drafts-dir", str(ROOT / "outputs" / "voice-producer-drafts"),
        "--runtime-root", str(ROOT / "voice-producer" / "models" / "runtime"),
        "--source-model-dir", str(ROOT / "voice-producer" / "models" / "checkpoints"),
        "--workspace-dir", str(ROOT / "outputs" / "voice-workbench"),
        "--output-dir", str(ROOT / "outputs" / "validation-web"),
        "--port", str(args.port), "--cpu-threads", str(args.cpu_threads),
    ])


if __name__ == "__main__":
    main()
