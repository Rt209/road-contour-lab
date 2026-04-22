# Road Contour Laboratory

道路輪廓擷取實驗專案，使用 `Python + OpenCV` 實作傳統影像處理流程，從輸入道路影像中擷取候選道路區域並輸出最終 contour 疊圖。

## 功能特色

- Sobel 特徵擷取
- LBP 特徵擷取
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
└─ src/
   ├─ preprocessing/
   ├─ features/
   ├─ segmentation/
   ├─ contour/
   ├─ pipeline/
   └─ utils/
```

## 處理流程

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

## 環境準備

以下示範使用 **Windows CMD**。

### 1. 進入專案資料夾

```cmd
cd /d C:\Users\GIGABYTE\RoadContour\road-contour-lab
```

### 2. 建立虛擬環境

```cmd
python -m venv .venv
```

### 3. 啟用虛擬環境

```cmd
.venv\Scripts\activate.bat
```

### 4. 安裝相依套件

```cmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 執行方式

將道路影像放入 `data/input/`，再執行下列指令：

```cmd
python main.py --input data/input/road.jpg --output_dir data/output
```

## 輸出說明

執行完成後，`data/output/` 資料夾內只會有一張結果圖。

輸出檔名會依輸入影像名稱自動產生，例如：

```text
data/output/road-result-時間戳記.jpg
```

## 參數設定

預設參數位置：

```text
configs/default_config.py
```

可依需求調整的參數包含：

- blur kernel size
- Sobel kernel size
- LBP radius / n_points
- fusion weights
- candidate threshold
- seed threshold
- BFS connectivity
- contour min area

## Python 版本

建議使用：

```text
Python 3.11
```

## 核心檔案

- `main.py`: 命令列入口
- `src/pipeline/road_contour_pipeline.py`: 主流程
- `configs/default_config.py`: 預設參數
