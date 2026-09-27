"""자동 채점기: pytest 를 돌려 항목별 점수를 계산합니다.

사용: python tests/score.py            (이 저장소의 tests/ 로 채점)
      python tests/score.py DIR [DIR…]  (교수자: 공식 테스트 폴더로 채점)
테스트 이름 test_<항목>__<설명> 에서 <항목>이 채점 단위이며, 항목 안의 테스트가 모두 통과해야 점수를 받습니다.
"""
import json, os, sys
import pytest

POINTS = {  # 항목: (점수, 설명)
    'p1_conv_out':       (5,  'Part 1  conv_out 출력 크기 공식'),
    'p1_count_params':   (5,  'Part 1  count_params'),
    'p1_smallcnn':       (3,  'Part 1  SmallCNN 출력 모양'),
    'p1_smallcnn_limit': (2,  'Part 1  SmallCNN 파라미터 20만 이하'),
    'p1_evaluate':       (10, 'Part 1  evaluate 정확도·eval 모드'),
    'p2_grid':           (10, 'Part 2  YOLO 격자 크기 = conv_out 반복'),
    'p2_num_pred':       (5,  'Part 2  출력 예측 개수 (8400)'),
    'p2_stride2_det':    (5,  'Part 2  탐지 모델의 stride 2 conv'),
    'p2_stride2_cls':    (5,  'Part 2  분류 모델의 stride 2 conv'),
    'p2_count_params_yolo': (5, 'Part 2  YOLOv8n 파라미터 수'),
    'p3_json':           (5,  'Part 3  results.json 완성'),
    'p3_consistent':     (5,  'Part 3  결과와 코드 일치'),
    'p3_cnn_acc':        (5,  'Part 3  SmallCNN 시험 정확도 ≥ 0.90'),
    'p3_yolo_cls':       (3,  'Part 3  YOLO 분류 학습 결과'),
    'p3_detection':      (2,  'Part 3  내 사진 탐지 결과'),
    'p3_images':         (5,  'Part 3  그림 2장'),
    'p4_report':         (5,  'Part 4  보고서 작성 여부 (내용 15점은 교수자 채점)'),
}


class Collector:
    def __init__(self):
        self.results = {}

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when == 'setup' and report.outcome != 'passed'):
            name = report.nodeid.split('::')[-1].split('[')[0]
            group = name[len('test_'):].split('__')[0]
            ok = report.outcome == 'passed'
            self.results.setdefault(group, []).append((report.nodeid, ok))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sys.argv[1:] or [here]
    col = Collector()
    pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=line', *dirs], plugins=[col])
    rows, total, maxi = [], 0, 0
    for g, (pts, desc) in POINTS.items():
        res = col.results.get(g, [])
        ok = bool(res) and all(r[1] for r in res)
        got = pts if ok else 0
        total += got; maxi += pts
        rows.append((desc, got, pts, sum(r[1] for r in res), len(res)))
    lines = ['| 항목 | 점수 | 통과한 테스트 |', '| --- | --- | --- |']
    lines += [f'| {d} | {g} / {p} | {a} / {n} |' for d, g, p, a, n in rows]
    lines.append(f'| **자동 채점 합계** | **{total} / {maxi}** | |')
    table = '\n'.join(lines)
    print('\n' + table)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write('## 자동 채점 결과\n\n' + table + '\n\n보고서 내용 15점은 교수자가 따로 채점합니다.\n')
    with open(os.environ.get('SCORE_JSON', 'score.json'), 'w', encoding='utf-8') as f:
        json.dump({'total': total, 'max': maxi, 'items': {d: g for d, g, *_ in rows}}, f, ensure_ascii=False)
    sys.exit(0 if total == maxi else 1)


if __name__ == '__main__':
    main()
