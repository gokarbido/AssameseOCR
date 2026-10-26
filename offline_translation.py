"""Local CPU INT8 IndicTrans2 inference; never downloads or sends OCR text."""
import argparse
import json
import os
from pathlib import Path
import re

import numpy as np


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / 'models/translation/indictrans2-int8'


class OfflineTranslator:
    def __init__(self, model_dir=None, max_tokens=64):
        root = Path(model_dir or os.environ.get('TRANSLATION_MODEL_DIR', DEFAULT_MODEL_DIR))
        required = ('encoder_model.onnx', 'encoder_model.onnx.data', 'decoder_model.onnx',
                    'decoder_with_past_model.onnx', 'decoder_shared.onnx.data',
                    'tokenizer_src.json', 'tokenizer_tgt.json', 'tokenizer_meta.json',
                    'generation_config.json')
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RuntimeError('Offline translation model missing. Run '
                               'python3 scripts/download_translation_model.py on the host, '
                               f'then restart OCR. Model folder: {root}')
        # Lazy imports keep OCR usable when optional translation dependencies fail.
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        from indicnlp.tokenize.indic_tokenize import trivial_tokenize
        from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
        from sacremoses import MosesDetokenizer, MosesPunctNormalizer

        self.normalize = IndicNormalizerFactory().get_normalizer('as').normalize
        self.tokenize = trivial_tokenize
        self.transliterate = UnicodeIndicTransliterator.transliterate
        self.punctuation = MosesPunctNormalizer()
        self.detokenize = MosesDetokenizer(lang='en').detokenize
        self.source = Tokenizer.from_file(str(root / 'tokenizer_src.json'))
        self.target = Tokenizer.from_file(str(root / 'tokenizer_tgt.json'))
        self.meta = json.loads((root / 'tokenizer_meta.json').read_text(encoding='utf-8'))
        generation = json.loads((root / 'generation_config.json').read_text(encoding='utf-8'))
        self.start_id = generation.get('decoder_start_token_id', 2)
        self.eos_id = generation.get('eos_token_id', 2)
        self.max_tokens = max_tokens
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.encoder, self.decoder, self.cached = [
            ort.InferenceSession(str(root / name), sess_options=options,
                                 providers=['CPUExecutionProvider'])
            for name in ('encoder_model.onnx', 'decoder_model.onnx', 'decoder_with_past_model.onnx')]

    def _prepare(self, text):
        # Specialize the upstream IndicTrans preprocessing to Assamese -> English.
        text = self.punctuation.normalize(text)
        text = text.translate(str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789'))
        protected = []

        def protect(match):
            protected.append(match.group())
            return f'<ID{len(protected)}>'

        text = re.sub(r'https?://[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\d+(?:[.,:/-]\d+)+', protect, text)
        text = ' '.join(self.tokenize(self.normalize(text.strip()), 'as'))
        text = self.transliterate(text, 'as', 'hi').replace(' ् ', '्')
        return 'asm_Beng eng_Latn ' + text.strip(), protected

    def translate(self, text):
        if not text.strip():
            return ''
        prepared, protected = self._prepare(text)
        encoded = self.source.encode(prepared)
        if len(encoded.ids) > 256:
            raise RuntimeError('Detected text is too long for offline translation; use a shorter text region.')
        ids = [[token if token < self.meta['src_dict_size'] else self.meta['unk_id'] for token in encoded.ids]]
        mask = np.asarray([encoded.attention_mask], dtype=np.int64)
        hidden = self.encoder.run(['last_hidden_state'],
                                  {'input_ids': np.asarray(ids, dtype=np.int64), 'attention_mask': mask})[0]
        feed = {'input_ids': np.array([[self.start_id]], dtype=np.int64),
                'encoder_hidden_states': hidden, 'encoder_attention_mask': mask}
        session = self.decoder
        generated = []
        for _ in range(self.max_tokens):
            values = session.run(None, feed)
            token = int(np.argmax(values[0][0, -1]))
            if token == self.eos_id:
                break
            generated.append(token)
            # Exported cache names share the suffix after present./past_key_values.
            outputs = {node.name: value for node, value in zip(session.get_outputs(), values)}
            feed = {'input_ids': np.array([[token]], dtype=np.int64), 'encoder_attention_mask': mask}
            for node in self.cached.get_inputs():
                if node.name.startswith('past_key_values.'):
                    feed[node.name] = outputs[node.name.replace('past_key_values.', 'present.', 1)]
            session = self.cached
        else:
            raise RuntimeError('Offline translation reached its length limit; use a shorter text region.')
        safe = [token if token < self.meta['tgt_dict_size'] else self.meta['unk_id'] for token in generated]
        english = self.target.decode(safe, skip_special_tokens=True)
        for number, original in enumerate(protected, 1):
            english = re.sub(r'[<\[]\s*ID\s*' + str(number) + r'\s*[>\]]', lambda _: original,
                             english, flags=re.IGNORECASE)
        return self.detokenize(english.split())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate Assamese to English locally with INT8 ONNX')
    parser.add_argument('text')
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    args = parser.parse_args()
    print(OfflineTranslator(args.model_dir).translate(args.text))
