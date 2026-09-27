import os, sys, functools
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.environ.setdefault('YOLO_VERBOSE', 'False')


@functools.lru_cache(maxsize=None)
def _load(name):
    from ultralytics import YOLO
    model = YOLO(name).model.eval()
    for p in model.parameters():      # ultralytics 는 추론용으로 불러오며 requires_grad 를 꺼 둔다
        p.requires_grad = True
    return model


@pytest.fixture(scope='session')
def yolo_det():
    """YOLOv8n 탐지 모델 (torch nn.Module)"""
    return _load('yolov8n.pt')


@pytest.fixture(scope='session')
def yolo_cls():
    """YOLOv8n 분류 모델 (torch nn.Module)"""
    return _load('yolov8n-cls.pt')


def actual_grids(model, imgsz):
    """탐지 헤드(Detect)로 들어가는 특징 맵 3개의 실제 한 변 크기와 출력 예측 개수"""
    import torch
    seen = {}
    h = model.model[-1].register_forward_hook(lambda m, inp, out: seen.__setitem__('x', inp[0]))
    with torch.no_grad():
        out = model(torch.zeros(1, 3, imgsz, imgsz))
    h.remove()
    out = out[0] if isinstance(out, (list, tuple)) else out
    return [t.shape[-1] for t in seen['x']], out.shape[-1]
