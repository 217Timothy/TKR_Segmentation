# TKR 傷口分割 inference

輸入一張照片或一個圖片資料夾後，程式會：

1. 計算 OOD score，判斷是不是模型可接受的 TKR 術後傷口照。
2. 對接受的照片產生傷口 segmentation mask。
3. 儲存 mask、疊圖、去背結果及 JSON 數值。

非 TKR／模型不接受的照片會顯示 `REJECT_OOD`，輸出的 mask 會是空白。

## 1. 放置模型

從 GitHub 下載程式後，將模型放在：

```text
tkr_inference/weights/tkr_seg_validity_gate_b3_256.pt
```

完整目錄應如下：

```text
TKR_Segmentation/
├── tkr_inference/
│   ├── pipeline.py
│   ├── cli.py
│   └── weights/
│       └── tkr_seg_validity_gate_b3_256.pt
├── requirements.txt
└── run.sh
```

模型 SHA-256：

```text
08b8080f7dd14f3a316d6de3e94305d654a63b58bf098abf6a88db3cb06c8572
```

## 2. 安裝

需要 Python 3.10 以上：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
./run.sh verify
```

看到 `package verification: PASS` 代表模型位置、模型版本與套件都正確。

## 3. Inference

單張照片：

```bash
./run.sh infer \
  --input /path/to/image.jpg \
  --output /path/to/new_output_dir
```

整個資料夾，包含子資料夾：

```bash
./run.sh infer \
  --input /path/to/images \
  --output /path/to/new_output_dir
```

程式預設會依序選擇 CUDA、Apple MPS、CPU。需要指定裝置時：

```bash
./run.sh infer \
  --input /path/to/image.jpg \
  --output /path/to/new_output_dir \
  --device cpu
```

每次請使用一個尚未存在的新 output 資料夾，以免蓋掉舊結果。

## 4. 輸出內容

每張圖片會建立一個結果資料夾：

```text
new_output_dir/
├── summary.json
└── image_name_xxxxxxxx/
    ├── result.json
    ├── mask.png
    ├── overlay.png
    └── masked.png
```

| 檔案 | 內容 |
|---|---|
| `result.json` | 接受／拒絕、OOD score、門檻、bbox 等數值 |
| `mask.png` | 原圖大小的二元傷口 mask |
| `overlay.png` | 在原圖上以紅色標出傷口 |
| `masked.png` | 只保留 mask 內部，其他位置為黑色 |
| `summary.json` | 整批圖片的總數與每張結果 |

`result.json` 的重要欄位：

| 欄位 | 意義 |
|---|---|
| `accepted` | 是否通過 TKR validity/OOD gate |
| `decision` | `REJECT_OOD`、`ACCEPT_TKR_WOUND_FOUND` 或 `ACCEPT_TKR_NO_WOUND_MASK` |
| `ood_score` | 越高越不像模型可接受的 TKR 照片 |
| `ood_threshold` | 固定拒絕門檻；目前為 `0.38371683454513555` |
| `mask_non_empty` | 最終 mask 是否包含傷口區域 |
| `bbox_xyxy` | 傷口 mask 的外接矩形；沒有 mask 時為 `null` |

判斷規則：

```text
ood_score >= ood_threshold  → REJECT_OOD，mask 歸零
ood_score <  ood_threshold  → 接受，再輸出 segmentation mask
```
