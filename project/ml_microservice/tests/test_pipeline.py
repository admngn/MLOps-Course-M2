from src.pipeline import build_pipeline

def test_build_pipeline():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    steps = dict(pipe.named_steps)
    assert "pre" in steps
    assert "model" in steps
