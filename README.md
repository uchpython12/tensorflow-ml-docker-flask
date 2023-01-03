# Python Flask 架設機器學習 API
此範例使用鳶尾花朵資料集進行 `XGBoost` 分類器模型訓練。將模型儲存起來，並使用 Flask 建置 API 介面提供輸入值預測。

## Getting Started
### Clone Project
你可以在本機直接使用 git 指令 clone 此專案並執行。

```
git clone https://github.com/uchpython12/tensorflow-ml-docker-flask
cd https://github.com/uchpython12/tensorflow-ml-docker-flask
```

### Docker Build Image
#### Docker自動打包image,本地端安裝請跳至Installation
docker run過後可直接訪問 [localhost:3000](http://localhost:3000/).

```
docker build -t tensorflow-ml-docker-flask .
docker run -p 3000:3000 tensorflow-ml-docker-flask
```

### Installation
此專案下載至桌面後，使用以下指令安裝必要套件。

```
pip install -r requirements.txt
```

### Running the Project
套件安裝成功後，即可開始執行本專案。

```
python run.py
```

running locally! Your app should now be running on [localhost:3000](http://localhost:3000/).
