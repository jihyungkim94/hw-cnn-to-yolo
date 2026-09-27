"""과제: CNN에서 YOLO로 — "YOLO도 결국 CNN이다"를 코드로 확인하기

이 파일의 TODO 를 채워 GitHub 에 올리면 자동 채점이 돌아갑니다.
- 함수 이름, 인자, 반환 형식은 바꾸지 마세요. (채점 테스트가 이 이름으로 불러 씁니다)
- `raise NotImplementedError` 줄을 지우고 코드를 쓰면 됩니다.
- import 는 torch, torch.nn 만 쓰세요. (ultralytics 는 이 파일에서 import 하지 않습니다)
"""
import torch
import torch.nn as nn


# ══════════════════════════ Part 1. CNN 기초 (4장) ══════════════════════════

def conv_out(n, k, s=1, p=0):
    """합성곱(또는 풀링) 한 번을 지난 뒤의 한 변 크기를 반환한다.

    n: 입력 한 변 크기, k: 커널 크기, s: 스트라이드, p: 패딩
    예) conv_out(28, 3, 1, 1) == 28,  conv_out(28, 3, 2, 1) == 14,  conv_out(7, 2, 2, 0) == 3
    """
    # TODO: 4장에서 배운 출력 크기 공식을 한 줄로 쓰세요.
    raise NotImplementedError


def count_params(model):
    """모델의 학습 가능한(requires_grad=True) 파라미터 개수를 정수로 반환한다.

    예) count_params(nn.Linear(3, 2)) == 8   (가중치 6 + 편향 2)
    """
    # TODO
    raise NotImplementedError


class SmallCNN(nn.Module):
    """Fashion-MNIST 분류기. 입력 (B, 1, 28, 28) → 출력 (B, 10).

    아래는 교재 [프로그램 4-2] 모델 그대로입니다. 이대로도 테스트는 통과하지만
    Part 3 에서 시험 정확도 90% 이상을 받아야 하므로 구조를 직접 개선하세요.
    조건: 출력 모양 (B, 10), 파라미터 200,000개 이하.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # → (B, 8, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)                           # → (B, 8, 14, 14)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # → (B, 16, 14, 14)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """loader 전체에 대한 정확도(맞힌 개수 / 전체 개수, 0~1 사이 실수)를 반환한다.

    - 평가 중에는 model.eval() 상태여야 한다.
    - 끝나면 모델을 원래 상태(train 이었으면 train)로 되돌린다.
    - loader 는 (x, y) 배치를 내놓는다. 예측은 model(x).argmax(dim=1).
    """
    # TODO
    raise NotImplementedError


# ══════════════════════════ Part 2. YOLO 안의 CNN (5장) ══════════════════════════
# YOLOv8n 백본의 0·1·3·5·7번 층은 4장에서 배운 nn.Conv2d(kernel_size=3, stride=2, padding=1) 입니다.
# 이 conv 를 지날 때마다 한 변이 절반이 되고, 3·4·5번째 conv 를 지난 특징 맵(P3·P4·P5)에서
# 물체를 찾습니다. README 의 그림을 먼저 보세요.

def yolo_grid_sizes(imgsz):
    """YOLOv8 탐지 헤드 3개(P3, P4, P5)의 격자 한 변 크기를 [P3, P4, P5] 리스트로 반환한다.

    반드시 위의 conv_out 을 이용해 계산할 것. (숫자를 외워서 쓰면 숨은 테스트에서 틀립니다)
    예) yolo_grid_sizes(640) == [80, 40, 20]
    """
    # TODO
    raise NotImplementedError


def yolo_num_predictions(imgsz):
    """YOLOv8 탐지 모델 출력의 예측(후보 상자) 개수를 반환한다.

    YOLOv8n 에 640×640 을 넣으면 출력 모양이 (1, 84, 8400) 이다. 이 8400 이 어디서 오는지
    yolo_grid_sizes 로 계산하라.
    """
    # TODO
    raise NotImplementedError


def find_stride2_convs(model):
    """model 안의 모든 nn.Conv2d 중 stride 가 (2, 2) 인 것을 등장 순서대로 리스트로 반환한다.

    힌트: model.modules() 는 모든 하위 모듈을 차례로 내놓는다. isinstance(m, nn.Conv2d), m.stride
    """
    # TODO
    raise NotImplementedError
