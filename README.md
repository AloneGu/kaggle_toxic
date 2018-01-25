# kaggle_toxic

get muse fasttext word embedding from https://github.com/facebookresearch/MUSE 

get glove embedding from https://nlp.stanford.edu/projects/glove/

# my rules

* generate resample features and use xgb to ensemble

* change ratio in base and xgb , new trans (expit(logit(x)-0.5))
