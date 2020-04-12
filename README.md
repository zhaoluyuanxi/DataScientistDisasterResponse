# Readme


优达学诚数据科学家纳米学位灾害应对通道


## 内容列表

- [项目说明](#项目说明)
- [包含内容](#包含内容)
- [操作手册](#操作手册)



## 项目说明

此项目实现的主要功能为提供一个网址,给网址中输入求救信息，网址会给求救信息进行分类，区分此求救是哪种求助，比如缺水等.


## 包含内容

1. 数据清理，会对过往已有信息进行清洗
2. 模型训练，对以往已有标签分类进行训练
3.通过FLASK进行部署

## 操作手册

 - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
 2. Run the following command in the app's directory to run your web app.
    `python run.py`


