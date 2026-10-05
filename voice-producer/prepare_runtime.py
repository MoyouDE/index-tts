"""Explicit preparation of the two audition runtimes; never called by start.py."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from indextts.runtime.assets import main as verify_assets
    from indextts.runtime.model_export import export_runtime_model, verify_runtime_model
    from indextts.runtime.profiles import PROFILES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model-dir', default=str(ROOT/'voice-producer/models/checkpoints'))
    parser.add_argument('--output-dir', default=str(ROOT/'voice-producer/models/runtime'))
    parser.add_argument('--profile', choices=PROFILES, action='append')
    args = parser.parse_args()
    if Path(args.source_model_dir).resolve() == ROOT/'voice-producer/models/checkpoints':
        verify_assets(['verify','--root',str(ROOT/'voice-producer/models'),'--lock',str(ROOT/'voice-producer/assets.lock.json')])
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    for profile in args.profile or PROFILES:
        destination = root/profile
        if destination.exists():
            verify_runtime_model(destination)
            print('Verified:', destination, flush=True)
            continue
        staging = root/('.prepare-'+uuid.uuid4().hex)
        try:
            export_runtime_model(args.source_model_dir, staging, profile=profile)
            verify_runtime_model(staging)
            os.rename(staging,destination)
            print('Prepared:', destination, flush=True)
        finally:
            if staging.exists() and staging.resolve().parent == root:
                shutil.rmtree(staging)


if __name__ == '__main__':
    main()
