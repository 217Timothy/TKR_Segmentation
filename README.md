# TKR 傷口分割與影像有效性判斷

推論套件 **2.3.0**（2026-11-09）。使用 **512×512** EfficientNet-B3／U-Net／SCSE 模型，先判斷影像是否可接受為 TKR 術後傷口照，再決定是否保留傷口 mask。分割底座維持原版權重，validity head 使用這次接續微調的最佳 epoch 1。

本 repository 只提供推論程式，不含訓練資料、照片、模型權重或下游分類功能。模型由提供者透過 Google Drive 私下分享；本專案不會自動下載權重。

## 安裝與模型檔

需要 Python 3.10 或以上。建議使用全新的虛擬環境。

```bash
git clone https://github.com/217Timothy/TKR_Segmentation.git
cd TKR_Segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows 使用 `python -m venv .venv`，PowerShell 啟用方式為 `.venv\Scripts\Activate.ps1`。

從提供者的 Google Drive 下載 **`tkr_seg_validity_gate_b3_512_20261109.pt`**，放到：

```text
tkr_inference/weights/tkr_seg_validity_gate_b3_512_20261109.pt
```

請使用這個完整檔名。舊256px或前一版512px權重不能替代這一份；更新程式後也要更新模型檔。

```bash
./run.sh verify
```

Windows 執行 `python verify_package.py`。看到 `package verification: PASS` 代表校驗碼、版本、模型資訊及基本推論檢查通過。校驗碼不符時，請重新取得配對的模型檔，不要修改 manifest 來略過檢查。

## 推論

單張圖片：

```bash
./run.sh infer --input /path/to/photo.jpg --output outputs/photo_run1
```

整個資料夾（包含子資料夾）：

```bash
./run.sh infer --input /path/to/photos --output outputs/folder_run1
```

Windows 或直接使用 Python：

```bash
python -m tkr_inference.cli --input C:/photos --output outputs/folder_run1
```

每次請指定新的輸出資料夾，程式不會覆蓋舊結果。支援 jpg/jpeg/png/bmp/webp。預設自動使用 CUDA、Apple MPS 或 CPU；可加 `--device cpu`、`--device cuda` 或 `--device mps`。

每張圖會輸出原圖大小的：

- `mask.png`：二值傷口遮罩。
- `overlay.png`：傷口範圍疊圖。
- `masked.png`：遮罩範圍以外設為黑色的影像。
- `result.json`：判斷、分數、mask 是否非空、範圍座標。

整批結果位於 `summary.json`。`bbox_xyxy` 是最終遮罩的 `[x_min, y_min, x_max_exclusive, y_max_exclusive]`，空遮罩時為 null。不輸出 ROI crop。

三種判斷：

| decision | 意義 |
|---|---|
| `REJECT_OOD` | 判定不接受，最終 mask 清空 |
| `ACCEPT_TKR_WOUND_FOUND` | 接受，且找到非空傷口 mask |
| `ACCEPT_TKR_NO_WOUND_MASK` | 接受，但沒有非空傷口 mask |

接受影像不代表一定能找到傷口，也不是臨床診斷。

## Python API

```python
from tkr_inference import TKRSegmentationPipeline

pipeline = TKRSegmentationPipeline(device="auto")
result = pipeline.predict("/path/to/photo.jpg")
print(result.metadata())
mask = result.mask  # 原圖大小、uint8、值為0或1
```

同一個 pipeline 可重複處理多張照片。也可傳入 OpenCV 格式的 `HxWx3 BGR uint8` NumPy array。

## 這一版的固定推論設定

- 輸入：512×512 letterbox、ImageNet normalization，保留有效影像區域遮罩。
- OOD score：`1 - valid_tkr_probability`，越高代表越不像可接受的 TKR 影像。
- 接受規則：`ood_score < 0.2077719569206238`，相等或更高即拒絕。
- 門檻由 checkpoint 自動讀取；僅用 validation 校準，沒有用最後全資料篩查調整。
- 分割機率門檻：0.5；後處理 blur 9、closing 9、opening 5、min_area 200，保留最大區域，再還原原圖尺寸。
- 新權重檔案的 SHA-256 與版本資料見 `manifest.json`。

這是研究用模型。已執行的全資料篩查含訓練資料與歷史重用資料，且兩張修正案例已納入訓練；舊分割底座也接觸過部分 validation。因此不能把既有篩查當成獨立泛化證明，或保證所有未來 TKR／非 TKR 照片都判斷正確。
