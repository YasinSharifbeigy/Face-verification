import numpy as np
import pytest
from verify import FaceVerifier


def verifier_with_embeddings():
    verifier = FaceVerifier.__new__(FaceVerifier)
    class Model:
        def get_providers(self):
            return ["test"]
    verifier.model = Model()
    verifier.embedding = lambda value: value / np.linalg.norm(value)
    return verifier


def test_cosine_decision():
    verifier = verifier_with_embeddings()
    a = np.array([1., 0.])
    assert verifier.verify(a, a)["same_person"]
    assert verifier.verify(a, -a)["score"] == -1
    assert not verifier.verify(a, np.array([0., 1.]))["same_person"]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        verifier_with_embeddings().verify(None, None, threshold)
