import json
import os
from typing import Optional

from transformers.file_utils import requires_backends
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class VITSTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        pad_token="<pad>",
        unk_token="<unk>",
        do_phonemize=True,
        **kwargs,
    ):
        if do_phonemize:
            requires_backends(self, "g2pk")

            from g2pk import G2P

            self.g2p = G2P()
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)
        self.do_phonemize = do_phonemize
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)

    def _tokenize(self, text, **kwargs):
        text = text.strip()

        if self.do_phonemize:
            text = self.g2p(text)

        tokens = text.split()

        return tokens

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.get_vocab(), ensure_ascii=False))

        return vocab_file
