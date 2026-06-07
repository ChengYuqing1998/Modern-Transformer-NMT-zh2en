import tempfile
import unittest
from pathlib import Path

from tokenizer.build_tokenizer import (
    build_decoder_only,
    build_encoder_decoder,
)


class TokenizerBuildTest(unittest.TestCase):
    def test_builds_separate_and_unified_vocabularies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "sample.txt"
            corpus.write_text("你好\thello world\n世界\tworld\n", encoding="utf-8")
            sizes = build_encoder_decoder(corpus, temp_dir, 16)
            sizes.update(build_decoder_only(corpus, temp_dir, 16))

            self.assertGreater(sizes["source"], 4)
            self.assertGreater(sizes["target"], 4)
            self.assertGreater(sizes["unified"], sizes["source"])
            self.assertTrue((Path(temp_dir) / "source_tokenizer.pkl").exists())
            self.assertTrue((Path(temp_dir) / "target_tokenizer.pkl").exists())
            self.assertTrue((Path(temp_dir) / "unified_tokenizer.pkl").exists())


if __name__ == "__main__":
    unittest.main()
