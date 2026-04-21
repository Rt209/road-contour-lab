# Road Contour Laboratory

道路輪廓擷取實驗專案，使用 `Python + OpenCV` 實作傳統影像處理流程，從輸入道路影像中擷取候選道路區域並輸出最終 contour 疊圖。

## 方法說明

目前版本使用 `Method A`：

- Sobel 邊緣特徵
- LBP 紋理特徵
- Feature Fusion
- Candidate Mask
- Distance Transform
- BFS Region Growing
- Connected Components Filtering
- Contour Extraction

## 專案結構

```text
road-contour-lab/
├─ main.py
├─ README.md
├─ requirements.txt
├─ config_example.py
├─ configs/
├─ data/
│  ├─ input/
│  └─ output/
├─ src/
│  ├─ preprocessing/
│  ├─ features/
│  ├─ segmentation/
│  ├─ contour/
│  ├─ pipeline/
│  └─ utils/
└─ tests/
```

## 流程架構

```text
Input Image
  -> Grayscale
  -> Gaussian Blur
  -> Sobel Feature Extraction
  -> LBP Feature Extraction
  -> Feature Fusion
  -> Candidate Mask Thresholding
  -> Morphology Cleanup
  -> Distance Transform
  -> Seed Point Selection
  -> BFS Region Growing
  -> Connected Components Filtering
  -> Contour Extraction
  -> Contour Overlay Output
```

## 安裝方式

以下示範以 **Windows CMD** 為主。

### 1. 進入專案目錄

```cmd
cd /d C:\Users\GIGABYTE\RoadContour\road-contour-lab
```

### 2. 確認目前工作路徑

```cmd
cd
```

預期輸出：

```cmd
C:\Users\GIGABYTE\RoadContour\road-contour-lab
```

### 3. 建立虛擬環境

```cmd
python -m venv .venv
```

### 4. 啟用虛擬環境

```cmd
.venv\Scripts\activate.bat
```

### 5. 安裝套件

```cmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install pytest
```

### 6. 執行測試

```cmd
pytest tests -q
```

## 使用方式

先把道路影像放進 `data/input/`，再執行以下指令。

### 基本執行

```cmd
python main.py --input data/input/road.jpg --output_dir data/output
```

預設會輸出為：

```text
data/output/road-result.jpg
```

### 不輸出中間結果

```cmd
python main.py --input data/input/road.jpg --output_dir data/output --no_intermediate
```

### 自訂門檻參數

```cmd
python main.py --input data/input/road.jpg --output_dir data/output --threshold_candidate 80 --min_area 200
```

### 查看說明

```cmd
python main.py --help
```

## 輸出檔案

執行完成後，常見輸出如下：

- `sobel_magnitude.png`
- `lbp_map.png`
- `fused_feature_map.png`
- `candidate_mask_raw.png`
- `candidate_mask.png`
- `distance_transform.png`
- `seed_visualization.png`
- `bfs_region_mask.png`
- `final_region.png`
- `road-result.jpg`

最終結果重點檔案：

```text
data/output/road-result.jpg
```

## 參數設定

預設設定集中在：

```text
configs/default_config.py
```

可調整項目包含：

- blur kernel size
- Sobel kernel size
- LBP radius / n_points
- fusion weights
- candidate threshold
- seed threshold
- BFS connectivity
- contour min area

## 測試方式

### 執行全部測試

```cmd
pytest tests -q
```

### 執行單一測試

```cmd
python tests\test_sobel.py
python tests\test_lbp.py
python tests\test_distance_seed.py
python tests\test_pipeline_smoke.py
```

## Python 版本

建議使用：

```text
Python 3.11
```

目前 `requirements.txt` 已額外處理 Python 3.13 相容版本，因此若你的機器只有 Python 3.13，也可以直接安裝。

## 主要入口

- `main.py`: 命令列入口
- `src/pipeline/road_contour_pipeline.py`: 主流程
- `configs/default_config.py`: 預設參數
- `tests/`: 基本測試
