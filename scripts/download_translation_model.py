"""One-time verified download. No model code is downloaded or executed."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REPOSITORY = 'hari31416/indictrans2-indic-en-dist-200M-ONNX-int8'
REVISION = 'a7ecdc0ba73148c6e1d051ca372332471d8aff4f'
FILES = {
    'encoder_model.onnx': '79dd1c3d9f90f8bbd48effcc2d150616ac3a97525c8081481f14cd5095567ca6',
    'encoder_model.onnx.data': '6c402df00dfc45a69607e0231e1e5fd884508b19ca664836c94dd61bfa1ba137',
    'decoder_model.onnx': '845cdbaf0ee97d3c8ddff249a2e2d6e475d7ae1781157fcc6d0d651405bb8dd6',
    'decoder_with_past_model.onnx': 'bcc818db56bc681fab2d50ca31b99dbcb5f82abbce4dccc92946edf5906fa86e',
    'decoder_shared.onnx.data': 'abe939ce4d60bcbb9744790640188686c8af084083eb50d63bdbb7266a325d9a',
    'tokenizer_src.json': 'd15ba1265a61b3b66122aab74e888442093a2c711f51b0f9896108661e1d346c',
    'tokenizer_tgt.json': None,
    'tokenizer_meta.json': None,
    'generation_config.json': None,
    'config.json': None,
    'README.md': None,
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'models/translation/indictrans2-int8')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = args.output / name
        if destination.is_file() and (expected is not None and digest(destination) == expected):
            print(f'Verified: {name}', flush=True)
            continue
        print(f'Downloading {name} ...', flush=True)
        temporary = destination.with_name(destination.name + '.part')
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
        if expected is not None and digest(temporary) != expected:
            raise RuntimeError(f'Checksum mismatch: {name}; the incomplete file was not installed.')
        temporary.replace(destination)
    (args.output / 'SOURCE.txt').write_text(f'{REPOSITORY}\nRevision: {REVISION}\nLicense: MIT\n', encoding='utf-8')
    print(f'Offline model ready: {args.output}', flush=True)


if __name__ == '__main__':
    main()
