"""Expose official kernel modules without importing unrelated model/FA2 packages.

Creates only symlinks under --output. Does not install packages or edit sources.
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fa4', type=Path, required=True)
parser.add_argument('--nunchaku', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--svdquant-extension', type=Path)
args = parser.parse_args()
fa4, nunchaku = args.fa4.resolve(), args.nunchaku.resolve()
links = {
    'fa4/flash_attn/cute': fa4 / 'flash_attn/cute',
    'nunchaku-ops/nunchaku/ops': nunchaku / 'nunchaku/ops',
    'nunchaku-ops/nunchaku/utils.py': nunchaku / 'nunchaku/utils.py',
    'nunchaku-ops/nunchaku/lora/flux/packer.py': nunchaku / 'nunchaku/lora/flux/packer.py',
    'nunchaku-ops/nunchaku/lora/flux/utils.py': nunchaku / 'nunchaku/lora/flux/utils.py',
}
extensions = ([args.svdquant_extension.resolve()] if args.svdquant_extension
              else list((nunchaku / 'nunchaku').glob('_C*.so')))
for extension in extensions:
    links[f'nunchaku-ops/nunchaku/{extension.name}'] = extension
for name, source in links.items():
    target = args.output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source, f'{target} points to another source'
    else:
        target.symlink_to(source)
