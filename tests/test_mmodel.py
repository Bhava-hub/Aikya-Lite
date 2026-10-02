import torch

from model import FraudNet


def test_output_shape():
    assert FraudNet(input_dim=30)(torch.randn(8, 30)).shape == (8, 1)


def test_model_outputs_raw_logits_not_probabilities():
    # BCEWithLogitsLoss needs raw logits. If a sigmoid sneaks back into forward(),
    # this fails, because sigmoid(10) is about 0.99995, not 10.
    model = FraudNet(input_dim=30)
    model.fc3.weight.data.zero_()
    model.fc3.bias.data.fill_(10.0)
    out = model(torch.randn(4, 30))
    assert torch.allclose(out, torch.full((4, 1), 10.0))


def test_saved_weights_reload_identically(tmp_path):
    torch.manual_seed(0)
    a, b = FraudNet(30), FraudNet(30)
    torch.save(a.state_dict(), tmp_path / "m.pt")
    b.load_state_dict(torch.load(tmp_path / "m.pt"))
    x = torch.randn(5, 30)
    assert torch.allclose(a(x), b(x))