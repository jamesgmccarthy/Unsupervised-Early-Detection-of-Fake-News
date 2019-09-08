# Unsupervised-Early-Fake-News-Detection
This is a code repository for my thesis project aimed at distinguishing fake news articles from real news articles in an unsupervised early detection approach using clustering techniques. A secondary aim of my thesis was to investigate whether or not clustering the news articles in dataset into their respective topics and then fitting the clustering models on the each topic cluster would improve the overall clustering performance of the models. This was done because it was believed that articles written about Donald Trump for example would have different linguistic styles and semantic features than articles written about healthcare.

## Getting Started
A number of packages are required. This can be found in the requirements.txt file. To install these packages simply pip install the packages by entering the following command in a terminal window after having navigated to the folder where the python files are located. 
```
pip install -r requirements.txt
```

### Downloading the data
The dataset used in this thesis was the [FakeNewsNet] (https://github.com/KaiDMML/FakeNewsNet) dataset and can be downloaded following the instructions its repository. Once the data has been downloaded, which is unfortunately a time consuming processes, it must be saved to a folder named 'Dataset'. Once the dataset has been downloaded, a secondary dataset must be downloaded from https://www.dropbox.com/s/gho59cezl43sov8/FakeNewsNet-master.zip?dl=0 and must be saved in the same folder as the FakeNewsNet dataset. This dataset is only used to train the Doc2vec model used in the experiment.

### Running the experiments
Once the datasets have been downloaded, to run the whole experiment simply navigate to the location of the python files and enter the following in a terminal window 
```
python pipeline.py
```
This will run the whole experimental process of extracting the necessary data from the downloaded files, then it will preprocess it, then the doc2vec models will be trained and then the topic detection experiment will run and finally the fake news detection phase will run. 

