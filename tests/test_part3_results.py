"""Part 3. 같은 데이터, 두 모델 — Colab 에서 만든 results/ 를 검사합니다"""
import json, os
import pytest
import hw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results', 'results.json')


@pytest.fixture(scope='module')
def res():
    assert os.path.exists(RES), 'results/results.json 이 없습니다. Colab 노트북을 끝까지 실행해 만든 파일을 올리세요'
    with open(RES, encoding='utf-8') as f:
        return json.load(f)


def _nulls(d, path=''):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _nulls(v, f'{path}.{k}')
    elif d is None:
        yield path


def test_p3_json__complete(res):
    missing = list(_nulls(res))
    assert not missing, f'비어 있는 값이 있습니다: {missing}'
    for k in ['smallcnn', 'yolo_cls', 'yolo_det', 'environment']:
        assert k in res, f'{k} 항목이 없습니다'


def test_p3_consistent__params_match_code(res):
    real = sum(p.numel() for p in hw.SmallCNN().parameters())
    assert res['smallcnn']['params'] == real, (
        f"results.json 의 SmallCNN 파라미터({res['smallcnn']['params']})가 지금 hw.py 의 SmallCNN({real})과 다릅니다. "
        '모델을 고쳤다면 노트북을 다시 실행하세요')


def test_p3_cnn_acc__at_least_90(res):
    acc = res['smallcnn']['test_acc']
    assert 0.90 <= acc <= 1.0, f'SmallCNN 시험 정확도 {acc} — 0.90 이상이 목표입니다 (교재 모델은 약 0.89)'


def test_p3_yolo_cls__trained(res):
    y = res['yolo_cls']
    assert 0.70 <= y['top1'] <= 1.0, f"YOLO 분류 top1 {y['top1']} — 0.70 이상이어야 합니다"
    assert y['params'] >= 1_000_000 and y['epochs'] >= 1 and y['imgsz'] >= 32


def test_p3_detection__own_photo(res):
    assert res['yolo_det']['image'] != 'bus.jpg', '예시 사진(bus.jpg)이 아니라 직접 찍은 사진을 쓰세요'
    dets = res['yolo_det']['detections']
    assert len(dets) >= 1, '직접 찍은 사진에서 물체를 하나 이상 탐지해야 합니다'
    for d in dets:
        assert set(d) >= {'name', 'conf', 'box'} and len(d['box']) == 4


@pytest.mark.parametrize('name', ['predictions.png', 'detection.png'])
def test_p3_images__exist(name):
    path = os.path.join(ROOT, 'results', name)
    assert os.path.exists(path), f'results/{name} 이 없습니다'
    from PIL import Image
    with Image.open(path) as im:
        assert im.size[0] >= 200 and im.size[1] >= 100
