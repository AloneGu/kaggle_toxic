# kaggle_toxic

link: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

get muse fasttext word embedding from https://github.com/facebookresearch/MUSE 

get glove embedding from https://nlp.stanford.edu/projects/glove/

get crawl fasttext word embeding from https://www.kaggle.com/yekenot/pooled-gru-fasttext/data

## stacking

* train base NN models

* train base lr, mnb and other models

* combine with other features, use xgb or lgb to ensemble (10 fold, pub score 9871, pri score 9868)

* blend with other results (pub score 9873, pri score 9869, final rank 99/45xx, top 3%， toxic小分队)
