"""Part 2. YOLO 안의 CNN (5장) — 실제 YOLOv8n 모델과 비교합니다"""
import torch.nn as nn
import pytest
import hw
from conftest import actual_grids


@pytest.mark.parametrize('imgsz', [640, 320])
def test_p2_grid__matches_real_yolo(yolo_det, imgsz):
    real, _ = actual_grids(yolo_det, imgsz)
    assert list(hw.yolo_grid_sizes(imgsz)) == real, f'{imgsz} 입력의 실제 격자는 {real} 입니다'


@pytest.mark.parametrize('imgsz', [640, 320])
def test_p2_num_pred__matches_output(yolo_det, imgsz):
    _, real = actual_grids(yolo_det, imgsz)
    assert hw.yolo_num_predictions(imgsz) == real, f'{imgsz} 입력의 실제 출력은 {real}개 입니다'


def test_p2_stride2_det__yolov8n(yolo_det):
    got = hw.find_stride2_convs(yolo_det)
    want = [m for m in yolo_det.modules() if isinstance(m, nn.Conv2d) and tuple(m.stride) == (2, 2)]
    assert all(isinstance(m, nn.Conv2d) for m in got), 'nn.Conv2d 모듈을 담은 리스트를 반환하세요'
    assert len(got) == len(want) == 7 and all(a is b for a, b in zip(got, want))


def test_p2_stride2_cls__yolov8n_cls(yolo_cls):
    got = hw.find_stride2_convs(yolo_cls)
    assert len(got) == 5, f'분류 모델의 stride 2 conv 는 5개인데 {len(got)}개를 찾았습니다'


def test_p2_count_params_yolo__det(yolo_det):
    want = sum(p.numel() for p in yolo_det.parameters())
    assert want == 3_157_200
    assert hw.count_params(yolo_det) == want
