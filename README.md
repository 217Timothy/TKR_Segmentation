# TKR 傷口分割、validity gate 與 ROI

推論套件 **2.4.0**（2026-11-11）。預設使用最新的 targeted-repair epoch 1 模型：validity head 保留原本的 512px 補黑邊輸入與校準門檻；接受後，同一份權重的分割部分改用歷史 ROI 的直接縮放與後處理，輸出傷口周圍裁切的 `roi.jpg`。

本專案只做推論。模型權重、醫療影像、病患資料與推論輸出不放 Git；模型需私下分享。本次沒有重新訓練或變更權重。

## 安裝與權重

需要 Python 3.10 或以上。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows 使用 `.venv\Scripts\Activate.ps1`。將私下取得的 `tkr_seg_validity_gate_b3_512_20261109.pt` 放入 `tkr_inference/weights/`。模型檔與2.3.0相同，不需換檔；本次更新的是推論流程。

```bash
./run.sh verify
```

Windows 使用 `python verify_package.py`。驗證會檢查權重校驗碼、epoch、門檻、版本、目前預設流程與基本推論。`manifest.json` 分開記錄 checkpoint 保存的原後處理與目前實際採用的後處理，沒有改寫權重中的設定。

## 預設推論：最新 gate + 舊樣式 ROI

在專案根目錄執行：

```bash
./run.sh infer --input /path/to/photo.jpg --output outputs/photo_run1
./run.sh infer --input /path/to/photos --output outputs/folder_run1
```

程式會遞迴搜尋 jpg/jpeg/png/bmp/webp。輸出資料夾必須不存在，避免覆蓋。預設自動使用 CUDA、Apple MPS 或 CPU；可指定 `--device cpu`、`--device mps` 等。Windows 可執行 `python -m tkr_inference.cli`，參數相同。

每張影像的輸出：

| 檔案 | 內容 |
|---|---|
| `mask.png` | 原圖大小的二值傷口 mask，0/255 |
| `overlay.png` | 原圖大小的分割疊圖 |
| `masked.png` | 原圖大小、mask 外全黑的影像 |
| `roi.jpg` | 接受且 mask 非空時才產生：黑底傷口裁切圖，bbox 外留40像素，JPEG品質95 |
| `result.json` | gate分數、判斷、處理模式、mask與ROI資訊 |

整批資訊在 `summary.json`，包含模型 SHA-256、epoch、門檻、實際前後處理及 ROI 數量。`bbox_xyxy` 表示mask本身範圍；`roi_bbox_xyxy` 包含40像素邊距，座標均在原圖中，右／下界為exclusive。

被拒絕的影像保持空mask、全黑masked影像，不執行第二次分割，也不產生 `roi.jpg`。接受但mask為空，同樣不產生ROI；不使用舊程式的背景移除備援。下游分類只能讀取 `roi_available=true` 的結果。

## 預設流程的固定設定

`historical-roi` 為 `--model validity` 與 Python API 的預設流程：

1. **Gate**：等比例縮放＋補黑邊至512×512，ImageNet normalization及content mask保持不變；OOD score = 1 − valid-TKR probability，接受條件仍為 `ood_score < 0.2077719569206238`。
2. **接受後分割**：用同一份最新權重，將原圖直接縮成512×512；ImageNet normalization；sigmoid > 0.5。
3. **後處理**：Gaussian blur7（sigma0）、127再二值化、橢圓closing7、無opening、只保留面積≥200的最大外輪廓並填滿，再closing7。
4. **還原與ROI**：最近鄰還原mask到原圖尺寸，mask外設黑，再裁到bbox＋40px，存JPEG品質95。

兩種輸入各自進入同一模型的不同分支；接受的照片需要第二次encoder／分割運算，因此可能比以前單次共用運算慢。沒有另外載入六月舊權重，也沒有新訓練或門檻校準。

`raw_mask_non_empty` 指此次分割輸入在後處理前是否非空。預設模式對拒絕影像跳過分割，所以該欄為 `null`，並標記 `segmentation_executed=false`；不得把null當成raw空mask，或拿來計算負樣本raw non-empty rate。

## 切回11月9日的補黑邊分割流程

```bash
./run.sh infer --profile letterbox --input /path/to/photos --output outputs/letterbox_run1
```

此模式維持 gate 與 segmentation 共用512px letterbox輸入，以及checkpoint的blur9／closing9／opening5／min_area200後處理；輸出原圖大小的mask、overlay、masked影像，不產生裁切ROI。也保留拒絕影像的raw mask是否非空紀錄。

## Python API

```python
from tkr_inference import TKRSegmentationPipeline

pipeline = TKRSegmentationPipeline(device="auto")
result = pipeline.predict("/path/to/photo.jpg")
print(result.metadata())
mask = result.mask  # 原圖大小、uint8、0或1
if result.roi_bgr is not None:
    roi = result.roi_bgr  # 裁切的黑底ROI，OpenCV BGR uint8

# 使用變更前的單次letterbox流程：
previous = TKRSegmentationPipeline(device="auto", profile="letterbox")
```

同一pipeline可處理多張照片，也接受OpenCV格式的HxWx3 BGR uint8 NumPy array。

## 六月純分割模型：僅供明確指定的歷史比較

`--model legacy` 才會載入六月的 `outputs/checkpoints/tkr-finetune-v1/exp2/best.pt`，完全沒有 validity head。它的預設仍為letterbox，保留原本比較指令的行為；也可明確指定 `--profile historical-roi` 重現歷史直接縮放ROI。

若要檢查新舊分割權重在相同流程下是否一致，兩個指令必須指定相同profile，例如：

```bash
./run.sh infer --model validity --profile letterbox --input "data/raw/TKR_Knee/images" --output "outputs/validity_letterbox_new_run"
./run.sh infer --model legacy --profile letterbox --checkpoint "outputs/checkpoints/tkr-finetune-v1/exp2/best.pt" --input "data/raw/TKR_Knee/images" --output "outputs/legacy_letterbox_new_run"
```

純分割的accepted、OOD score及threshold為null，代表沒有執行TKR判斷。不要把預設historical-roi結果直接與先前letterbox結果當成「相同推論流程」比較。

## 驗證範圍

本次使用既有影像做流程回歸驗證，不是新獨立test；歷史ROI重現不代表分割準確度或臨床效能提升。既有全資料篩查含train、validation及重用comparison，兩張修正案例已納入train，舊分割底座也接觸過部分validation，不能宣稱獨立泛化。
