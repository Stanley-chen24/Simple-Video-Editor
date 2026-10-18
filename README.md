# 簡易影片剪輯器 (Simple Video Editor)

這是一個小型的桌面影片剪輯器，使用 **PyQt5**、**OpenCV**（預覽）與 **FFmpeg**（匯出）構建。支援匯入多部 `.mp4` 影片至素材庫、於時間軸上自由排列與剪輯、刪除不要的片段、進行畫面裁切比例調整，並將多段影片合併匯出為單一新的 `.mp4` 檔案。

## 1. 專案說明

本專案提供直覺的多影片剪輯與管理功能。第一部匯入的影片將決定專案的解析度與影格率，其餘影片會自動進行縮放與影格轉換對齊。支援畫面裁切不同解析度選項。

## 2. 功能列表

**多影片管理**：支援一次匯入多個 `.mp4` 檔案或直接拖曳至視窗中，並提供獨立的素材庫檢視各影片資訊。
* **時間軸序列剪輯**：
  * 各影片以不同顏色顯示。
  * 提供隨處分割（`S`）、標記入/出點剪輯（`I` / `O` + `Shift+S`）以及刪除片段[cite: 1]。
* **畫面裁切與縮放**：提供預設比例（16:9、1:1、4:3、9:16）與自訂裁切區域，並支援指定輸出解析度[cite: 1]。
* **匯出設定**：可選擇輸出路徑、自訂解析度，並支援「匯出時將音訊靜音 (Mute Audio on Export)」選項[cite: 1]。
* **自動處理**：解析度、FPS 不同的影片會自動對齊第一支影片；沒有聲音的影片以靜音補上

## 3. 使用方式

1. 安裝（需要 Python 3.8 以上）：

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # macOS / Linux：source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. 執行：

   ```bash
   python main.py                  # 啟動後按 Ctrl+O 匯入影片，或把檔案拖進視窗
   ```

常用快捷鍵：`Space` 播放／暫停、`←` `→` 逐格、`S` 分割、`I` / `O` 標記、`Del` 刪除、`Alt+←` / `Alt+→` 調整順序、`Ctrl+E` 匯出。

## 4. Test功能以及補充說明

### 測試 (Tests)

- **自動化測試**（可確認套件是否已安裝成功）：會自動產生測試影片進行實際剪接、裁切、合併與匯出，並自動檢驗畫面、聲音與長度是否正確。
- **安裝檢查**：安裝完成後先執行下列指令，若最後一行顯示「程式可正常執行!」，代表套件與 FFmpeg 皆安裝成功。

```bash
python -m unittest discover -s tests -t . -v
```

### 程式結構 (Project Structure)

- `main.py`：主程式
- `videoeditor/`：GUI窗口
  - `models.py`：剪輯資料與狀態模型
  - `exporter.py`：影片匯出與 FFmpeg 建構
  - `player.py` 與 `audio.py`：影音預覽與播放控制
  - `main_window.py`：主視窗與操作互動邏輯
  - `timeline_widget.py`：時間軸介面與軌道控制
  - `video_widget.py`：畫面預覽與裁切
  - `export_dialog.py`：匯出設定對話框與背景渲染
  - `crop.py` / `history.py` / `media.py`：裁切幾何、復原堆疊與檔案處理
- `tests/`：整合測試
