
# coding: utf-8

# In[1]:


# ref: https://www.kaggle.com/tilii7/tuned-logreg-oof-files

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
tr_ids = train[['id']]
train[class_names] = train[class_names].astype(np.int8)
target = train[class_names]

print(' Cleaning ...')
# PREPROCESSING PART
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

keys = [i for i in repl.keys()]

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


# In[2]:


train.head()


# In[3]:


test.head()


# In[4]:


print(' Part 1/2 of vectorizing ...')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

print(' Part 2/2 of vectorizing ...')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(4, 6),
    max_features=15000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features]).tocsr()
test_features = hstack([test_char_features, test_word_features]).tocsr()
print(train_features.shape,test_features.shape)

del all_text
del word_vectorizer
del char_vectorizer
del train_word_features
del train_char_features
del test_word_features
del test_char_features
import gc
gc.collect()


# In[5]:


target = target.values
print(target[:5])


# In[ ]:


from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

model = LogisticRegression(solver='sag')
sfm = SelectFromModel(model, threshold=0.2)


# In[ ]:


def kf_train(k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1001)
    train_pred, test_pred = np.zeros((159571,6)),np.zeros((153164,6))
    for j in range(6):
        train_x = sfm.fit_transform(train_features, target[:,j])
        print(train_x.shape)
        test_x = sfm.transform(test_features)
        fold_idx = 0
        for train_index, test_index in skf.split(train_x,target[:,j]):
            
            # data
            curr_x,curr_y = train_x[train_index],target[train_index][:,j]
            hold_out_x,hold_out_y = train_x[test_index],target[test_index][:,j]
            d_train = lgb.Dataset(curr_x, label=curr_y)
            d_valid = lgb.Dataset(hold_out_x, label=hold_out_y)
            watchlist = [d_train, d_valid]
            
            params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 16,
              'metric': 'auc',
              'data_random_seed': 2,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1}
            
            # train
            lgb_m = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=200,
                      valid_sets=watchlist,
                      early_stopping_rounds=20,
                      verbose_eval=50)
            
            hold_out_pred = lgb_m.predict(hold_out_x)
            curr_train_pred = lgb_m.predict(curr_x)
            print(fold_idx,log_loss(hold_out_y,hold_out_pred),log_loss(curr_y,curr_train_pred))
            print(roc_auc_score(hold_out_y,hold_out_pred),roc_auc_score(curr_y,curr_train_pred))
            fold_idx += 1
            
            train_pred[test_index,j] = list(hold_out_pred.flatten())
            y_test = lgb_m.predict(test_x)
            test_pred[:,j] += y_test
        print('=========',class_names[j])
    test_pred = test_pred/k
    return train_pred, test_pred

train_pred,test_pred = kf_train(5)


# In[ ]:


import pickle
with open('../features/lgb1_feat.pkl','wb') as fout:
    pickle.dump([train_pred,test_pred],fout)
print(test_pred[:5])

