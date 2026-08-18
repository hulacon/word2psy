"""ebind_text: attributes and naming offline; loading gated behind the
module actually being exercised (EBind pulls the full PE checkpoint)."""

import numpy as np
import torch

from word2psy.models.ebind_text import EMBED_DIM, EBindTextModel


class TestEBindTextModel:
    def test_attributes(self):
        model = EBindTextModel(device="cpu")
        assert model.name == "ebind_text"
        assert model.level == "chunk"
        # Must match viz2psy ebind / aud2psy ebind_audio exactly — psytwill
        # asserts checkpoint equality before pairing shared spaces.
        assert model.checkpoint == "encord-team/ebind-full"

    def test_column_naming_four_digit(self):
        model = EBindTextModel(device="cpu")

        class FakeModel:
            def forward(self, text):
                return {"text": torch.ones(text.shape[0], EMBED_DIM)}

        model.model = FakeModel()
        model._tokenizer = lambda t: torch.zeros(32, dtype=torch.long)
        results = model.predict_batch(["carrot", "market"])
        assert len(results) == 2
        keys = sorted(results[0])
        assert keys[0] == "ebind_text_0000"
        assert keys[-1] == "ebind_text_1023"
        assert len(keys) == EMBED_DIM

    def test_l2_normalized(self):
        model = EBindTextModel(device="cpu")

        class FakeModel:
            def forward(self, text):
                return {"text": torch.arange(1, EMBED_DIM + 1, dtype=torch.float32).repeat(text.shape[0], 1)}

        model.model = FakeModel()
        model._tokenizer = lambda t: torch.zeros(32, dtype=torch.long)
        vec = np.array(list(model.predict("carrot").values()))
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
