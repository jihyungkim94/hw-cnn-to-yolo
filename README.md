# 과제: CNN에서 YOLO로 — "YOLO도 결국 CNN이다"

4장에서 만든 CNN과 5장의 YOLO는 같은 부품으로 만들어져 있습니다. 이 과제에서는 그걸 코드로 직접 확인합니다.

- **배점** 100점 = 자동 채점 85점 + 보고서 내용 15점 (교수자 채점)
- **제출** 이 저장소에 커밋하면 끝. 마감 시각 전 마지막 커밋으로 채점합니다.
- **환경** 코드 수정은 GitHub 웹 편집기(또는 내 컴퓨터), 학습은 Google Colab (T4 GPU)

## 왜 CNN과 YOLO가 이어지는가

YOLOv8n 백본의 0·1·3·5·7번 층은 4장에서 배운 `nn.Conv2d(kernel_size=3, stride=2, padding=1)` 입니다.
이 conv를 지날 때마다 한 변이 절반이 되고, 3·4·5번째를 지난 특징 맵에서 물체를 찾습니다.

```
입력 640×640
 ├ Conv s2 → 320×320          (0번 층)
 ├ Conv s2 → 160×160          (1번 층)
 ├ Conv s2 →  80×80  ──▶ P3 격자  80×80 = 6400칸  (작은 물체)
 ├ Conv s2 →  40×40  ──▶ P4 격자  40×40 = 1600칸
 └ Conv s2 →  20×20  ──▶ P5 격자  20×20 =  400칸  (큰 물체)
                                 합계 8400칸 → 출력 모양 (1, 84, 8400)
```

4장의 출력 크기 공식 하나로 YOLO의 출력 모양을 설명할 수 있다는 것이 이 과제의 핵심입니다.

## 할 일

| Part | 내용 | 파일 | 점수 |
| --- | --- | --- | --- |
| 1. CNN 기초 | `conv_out`, `count_params`, `evaluate` 작성, `SmallCNN` 개선 | `hw.py` | 25 |
| 2. YOLO 안의 CNN | `yolo_grid_sizes`, `yolo_num_predictions`, `find_stride2_convs` 작성 | `hw.py` | 30 |
| 3. 같은 데이터, 두 모델 | Colab에서 내 SmallCNN과 YOLOv8n-cls를 Fashion-MNIST로 학습, 내 사진으로 YOLO 탐지 | `results/` | 25 |
| 4. 보고서 | 질문 4개에 답하기 | `REPORT.md` | 20 (자동 5 + 내용 15) |

Part 3의 목표: **SmallCNN 시험 정확도 0.90 이상** (교재 모델 그대로는 약 0.89), SmallCNN 파라미터 **200,000개 이하**.

## 진행 순서

1. **내 저장소 만들기** — 교수자가 알려 준 템플릿 저장소에서 초록색 **Use this template → Create a new repository**.
   이름은 `hw-cnn-to-yolo` 그대로, 공개 범위는 교수자 안내를 따릅니다. 만든 저장소 주소를 교수자가 안내한 곳에 제출하세요.
2. **`hw.py` 채우기** — 저장소에서 `hw.py` → 연필 아이콘(Edit) → TODO를 채우고 **Commit changes**.
   커밋할 때마다 **Actions** 탭에서 자동 채점이 돌고, 실행 결과 화면 아래 **Summary**에 항목별 점수표가 나옵니다(패키지 설치 때문에 몇 분 걸립니다).
   Part 3·4를 하기 전이라 처음에는 빨간 ✗가 정상입니다.
3. **Colab에서 학습** — 아래 주소의 `아이디`만 바꿔 브라우저에 붙여 넣으면 노트북이 열립니다.
   `https://colab.research.google.com/github/아이디/hw-cnn-to-yolo/blob/main/notebooks/train_colab.ipynb`
   첫 셀의 `REPO_URL`을 내 저장소 주소로 바꾸고 **런타임 → 모두 실행**. 중간에 사진 올리기 버튼이 나오면 **직접 찍은 사진**을 올리세요.
   끝나면 `results.zip`이 내려받아집니다.
4. **결과 올리기** — `results.zip`을 풀고, 저장소의 `results` 폴더에서 **Add file → Upload files**로
   `results.json`, `predictions.png`, `detection.png` 세 파일을 올려 커밋합니다.
5. **보고서** — `REPORT.md`를 열어 학번·이름과 Q1~Q4 답을 쓰고 커밋합니다.
6. **확인** — Actions 탭에 초록 ✓가 뜨면 자동 채점 85점 만점입니다.

## 규칙

- `tests/` 폴더는 고치지 마세요. 최종 채점은 교수자가 공식 테스트(숨은 테스트 포함)로 다시 돌립니다.
  공개 테스트만 통과하도록 숫자를 외워 쓴 코드는 숨은 테스트에서 0점이 됩니다.
- `hw.py`의 `SmallCNN`을 고쳤다면 **노트북을 다시 실행해 `results/`를 다시 올려야** 합니다.
  `results.json`의 파라미터 수가 지금 코드와 다르면 감점됩니다.
- 탐지 사진은 직접 찍은 사진을 쓰세요. 예시 사진(bus.jpg)을 그대로 내면 Part 3 탐지 점수를 받지 못합니다.
- 함수 이름·인자·반환 형식은 바꾸지 마세요.

## 내 컴퓨터에서 채점해 보기 (선택)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests/score.py
```

## 자주 생기는 문제

| 증상 | 해결 |
| --- | --- |
| Actions에 `NotImplementedError` | 그 함수의 `raise NotImplementedError` 줄을 지우고 코드를 쓰세요 |
| 노트북 첫 셀에서 `git clone` 실패 | `REPO_URL`이 내 저장소 주소인지, 저장소가 공개인지 확인 |
| `p3_consistent` 실패 | `hw.py`를 고친 뒤 노트북을 다시 돌리지 않았습니다 |
| Colab이 느림 | 런타임 유형이 T4 GPU인지 확인 |
