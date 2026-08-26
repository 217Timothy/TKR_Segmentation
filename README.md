# TKR segmentation + OOD inference package v2

這是一包可以獨立執行的 TKR 術後傷口 inference pipeline。它會直接用同一個模型完成兩件事：

1. 判斷輸入是否為模型可接受的 TKR 術後傷口照。
2. 對通過 gate 的影像產生傷口 mask、bbox 與 ROI，交給後面的 classification pipeline。

被判為非 TKR／超出已知分布的影像會回傳 `REJECT_OOD`、把 mask 歸零，且 `should_classify=False`。同學串接時只要遵守這個欄位，就不會把被拒絕的影像送進分類器。

目前版本固定使用 EfficientNet-B3 + SCSE U-Net segmentation decoder + image-level validity/OOD head，輸入解析度為 256 × 256。部署門檻為 `0.38371683454513555`，它只用 grouped validation TKR 影像校正，沒有用 locked test 挑 threshold。

> 本套件只供研究與原型驗證，不是醫療器材，也不能單獨用於臨床決策。

## 交付方式

- 私下交付的 ZIP：包含模型權重，解壓後可以直接安裝並 inference。
- GitHub：只放程式、文件與權重 SHA-256，不公開模型權重、病患影像或推論輸出。
- GitHub 版本使用者需把私下收到的權重放到 `tkr_inference/weights/tkr_seg_validity_gate_b3_256.pt`。

正確權重的 SHA-256 是：

```text
08b8080f7dd14f3a316d6de3e94305d654a63b58bf098abf6a88db3cb06c8572
```

## 最快開始

需要 Python 3.10 以上。第一次使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
./run.sh verify
```

處理單張圖片：

```bash
./run.sh infer --input /path/to/image.jpg --output /path/to/new_output_dir
```

遞迴處理整個資料夾：

```bash
./run.sh infer --input /path/to/images --output /path/to/new_output_dir
```

預設會依序選 CUDA、Apple MPS、CPU，也可以加上 `--device cpu` 或 `--device cuda:0`。輸出資料夾必須是尚未存在的新路徑，避免覆蓋舊結果。

## 接在 classification 前面

模型請在服務啟動時載入一次，不要每張照片重新載入：

```python
from tkr_inference import TKRSegmentationPipeline

segmenter = TKRSegmentationPipeline(device="auto")
result = segmenter.predict("example.jpg")

if not result.accepted:
    response = {
        "status": "rejected",
        "ood_score": result.ood_score,
    }
elif not result.should_classify:
    response = {"status": "accepted_but_no_wound_roi"}
else:
    # roi_bgr 是 OpenCV BGR ndarray
    classification_result = classifier.predict(result.roi_bgr)
    response = {
        "status": "classified",
        "classification": classification_result,
    }
```

完整範例在 `example_classification_integration.py`。如果 classifier 需要 RGB：

```python
import cv2

roi_rgb = cv2.cvtColor(result.roi_bgr, cv2.COLOR_BGR2RGB)
```

如果 classifier 要吃完整尺寸的去背影像，可使用 `result.masked_bgr`；如果由它自己裁切，可使用 `result.mask` 與 `result.bbox_xyxy`。

## 串接判斷規則

| `decision` | `accepted` | `should_classify` | 下游處理 |
|---|---:|---:|---|
| `REJECT_OOD` | false | false | 拒絕，不呼叫 classifier |
| `ACCEPT_TKR_NO_WOUND_MASK` | true | false | 接受為 TKR，但無可用 ROI，不呼叫 classifier |
| `ACCEPT_TKR_WOUND_FOUND` | true | true | 將 ROI 送入 classifier |

## Python 回傳欄位

| 欄位 | 意義 |
|---|---|
| `accepted` | `ood_score` 小於門檻，判為可接受 TKR 輸入 |
| `should_classify` | `accepted` 且最終 mask 非空；只有此值為真才呼叫 classifier |
| `decision` | 三種流程狀態，見上表 |
| `ood_score` | 越高越不像 valid TKR；大於等於門檻即拒絕 |
| `valid_tkr_probability` | `1 - ood_score`；是模型分數，不是臨床機率 |
| `mask` | 原圖大小、值為 0/1 的 `uint8` mask |
| `bbox_xyxy` | `(x1, y1, x2, y2)`，右下座標不包含在切片內 |
| `roi_bgr` | 含預設 40 px padding 的傷口 crop；無 mask 時為 `None` |
| `masked_bgr` | 原圖大小，mask 外為黑色 |
| `overlay_bgr` | 原圖大小的紅色 mask 疊圖 |

命令列每張圖會輸出 `result.json`、`mask.png`、`overlay.png`、`masked.png`；有 ROI 才會有 `roi.png`。整批結果另有 `summary.json`。

## Pipeline 順序

1. 保持長寬比 letterbox 到 256 × 256，並做 ImageNet normalization。
2. 共用 EfficientNet-B3 encoder 同時計算 validity/OOD 分數與 segmentation。
3. `ood_score >= 0.38371683454513555` 時將 mask 歸零並停止下游分類。
4. 通過 gate 才清理 mask、還原至原圖大小並取得 wound ROI。
5. 只有 `should_classify=True` 才把 ROI 送入 classification pipeline。

## 目前研究結果

- 部署 checkpoint：epoch 3。
- segmentation grouped validation：32 張、8 groups；Dice `0.9676`、IoU `0.9374`、recall `0.9694`、precision `0.9664`。
- gate grouped validation：349 張 TKR、54 groups，加上 190 張 validation OOD；TKR acceptance 與 non-TKR rejection 都是 100%，AUROC/AUPRC 都是 1.0，FPR@TPR95 為 0。
- 重複使用的比較集：245 張 TKR、46 groups與 250 張 OpenImages；TKR acceptance 與 non-TKR rejection 都是 100%，負樣本 raw/gated non-empty-mask rate 為 `65/250 -> 0/250`。
- 歷史 SurgWound 67 張：non-TKR rejection 100%，raw/gated non-empty-mask rate 為 `48/67 -> 0/67`。

這些數字只代表目前資料與門檻，不等於已證明能拒絕所有真實世界非 TKR 影像。完整模型說明、資料限制與正確解讀請看 `MODEL_CARD.md`。

## 檔案索引

- `tkr_inference/pipeline.py`：Python API、模型架構、前後處理。
- `tkr_inference/cli.py`：單張／資料夾命令列入口。
- `example_classification_integration.py`：接 classification 的最小範例。
- `manifest.json`：版本、門檻、評估摘要與權重 SHA-256。
- `MODEL_CARD.md`：用途、評估、限制與風險。
- `verify_package.py`：權重完整性與不含病患影像的 forward smoke test。
- `tkr_inference/weights/README.md`：私下放置權重的方式。
