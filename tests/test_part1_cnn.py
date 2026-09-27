"""Part 1. CNN 기초 (4장)"""
import torch
import torch.nn as nn
import pytest
import hw

CASES = [(28, 3, 1, 1), (28, 3, 1, 0), (28, 5, 1, 2), (28, 5, 1, 0), (28, 3, 2, 1),
         (28, 7, 2, 3), (28, 4, 2, 1), (7, 2, 2, 0), (640, 3, 2, 1), (33, 3, 2, 1)]


@pytest.mark.parametrize('n,k,s,p', CASES)
def test_p1_conv_out__matches_torch(n, k, s, p):
    real = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, n, n)).shape[-1]
    assert hw.conv_out(n, k, s, p) == real, f'conv_out({n},{k},{s},{p}) 가 틀렸습니다. 실제 크기는 {real}'


def test_p1_count_params__basic():
    assert hw.count_params(nn.Linear(3, 2)) == 8
    assert hw.count_params(nn.Conv2d(8, 16, 3)) == 16 * 8 * 9 + 16


def test_p1_count_params__textbook_cnn():
    class Textbook(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.fc = nn.Linear(16 * 14 * 14, 10)
    assert hw.count_params(Textbook()) == 32618


def test_p1_count_params__skips_frozen():
    m = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    for p in m[0].parameters():
        p.requires_grad = False
    assert hw.count_params(m) == 3, 'requires_grad=False 인 파라미터는 세지 않아야 합니다'


def test_p1_smallcnn__output_shape():
    y = hw.SmallCNN()(torch.zeros(4, 1, 28, 28))
    assert tuple(y.shape) == (4, 10)


def test_p1_smallcnn_limit__params():
    n = sum(p.numel() for p in hw.SmallCNN().parameters())
    assert n <= 200_000, f'SmallCNN 파라미터가 {n:,}개입니다. 200,000개 이하로 줄이세요'


class _Spy(nn.Module):
    """정해진 로짓을 내놓고, 호출될 때 train/eval 상태를 기록하는 가짜 모델"""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.modes = []

    def forward(self, x):
        self.modes.append(self.training)
        out = torch.zeros(x.shape[0], 3)
        out[torch.arange(x.shape[0]), x[:, 0].long()] = 1.0   # x 첫 값 = 예측 클래스
        return out


def _loader():
    pred = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    true = torch.tensor([0, 1, 1, 0, 2, 2, 0])            # 7개 중 5개 정답
    return [(pred[:4, None].float(), true[:4]), (pred[4:, None].float(), true[4:])]


def test_p1_evaluate__accuracy():
    acc = hw.evaluate(_Spy(), _loader())
    assert isinstance(acc, float) and abs(acc - 5 / 7) < 1e-9, f'정확도는 5/7 이어야 하는데 {acc} 입니다 (배치별 평균이 아닌 전체 개수 기준)'


def test_p1_evaluate__eval_mode_and_restore():
    m = _Spy().train()
    hw.evaluate(m, _loader())
    assert m.modes and not any(m.modes), '평가 중에는 model.eval() 상태여야 합니다'
    assert m.training, '평가가 끝나면 원래 train 상태로 되돌려야 합니다'
