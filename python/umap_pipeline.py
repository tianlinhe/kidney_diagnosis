import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()

import umap
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from helper import plot_points, plot_bar, plot_cmatrix


class pline:
    def __init__(self, freq_thres='None'):
        self.freq_thres=freq_thres
        

    def load_data(self,X,y):
        """load X and y"""
        self.X=X
        self.y=y
        self.X.fillna(0, inplace=True)
        self.y.fillna(0, inplace=True)
    def selectX_name(self,select_pep):
        """select certain peptides from X
        X is a list of column names
        """
        
        self.X=self.X[self.X.columns[self.X.columns.isin(select_pep)]]
    def logX(self):
        """log(X+1) transformation of peptide intensity"""
        self.X=np.log(self.X+1)
    
    def train_test_split(self,train_size):
        """train_test stratifed split
        default is 7:3"""

        # y if categories, needs to be transformed to number before fit
        self.le=preprocessing.LabelEncoder()
        self.y_num=self.le.fit_transform(self.y)

        self.Xtrain, self.Xtest, self.ytrain, self.ytest=train_test_split(self.X,
                                                    self.y_num, 
                                                    random_state=42,
                                                    train_size=0.7,
                                                    stratify=self.y_num)
        if self.freq_thres !='None':
            thres=len(self.Xtrain)*self.freq_thres
            self.Xtrain=self.Xtrain[self.Xtrain.columns[(self.Xtrain!=0).sum()>=thres]]
            self.Xtest=self.Xtest[self.Xtrain.columns]
        print ('Xtrain.shape',self.Xtrain.shape)
        print ('Xtest.shape',self.Xtest.shape)

        
    def pipeline_fit(self,
            n_neighbors=list(range(5,50,10)),
            min_dist=[0.001,0.01,0.1,0.5],
            cv=5):
        """fit the grid search umap-svm pipeline"""

        svc = SVC()
        mapper = umap.UMAP(random_state=456)
        pipeline = Pipeline([("umap", mapper), ("svc", svc)])
        params_grid_pipeline = {
            "umap__n_neighbors": list(range(5,50,10)),
            "umap__min_dist":[0, 0.001,0.01,0.1,0.5],
        }

        mapper = umap.UMAP(random_state=456)
        pipeline = Pipeline([("umap", mapper), ("svc", svc)])
        params_grid_pipeline = {
            "umap__n_neighbors": n_neighbors,
            "umap__min_dist": min_dist,
        }

        self.clf_pipeline = GridSearchCV(pipeline, params_grid_pipeline,cv=5)

        self.clf_pipeline.fit(self.Xtrain, self.ytrain)
    def get_accuracy(self):
        """print accuracy scores of train and test data"""

        print(
            "Accuracy on the train set with UMAP transformation: {:.3f}".format(
                self.clf_pipeline.score(self.Xtrain, self.ytrain)
            )
        )
        print(
            "Accuracy on the test set with UMAP transformation: {:.3f}".format(
                self.clf_pipeline.score(self.Xtest, self.ytest)
            )
        )
    def get_confusion_matrix(self,on,figout=None):
        """plot confusion matrix on train or test results"""
   
        if on=='test':
            ytest_pred =self.clf_pipeline.predict(self.Xtest)
            plot_cmatrix(self.ytest,ytest_pred,cmapper=self.le,
                    output1=figout
                    )
        elif on=='train':
            ytrain_pred=self.clf_pipeline.predict(self.Xtrain)
            plot_cmatrix(self.ytrain,ytrain_pred,cmapper=self.le,
                    output1=figout
                    )
        else:
            print ("please specify on='test' or on='train'")
            

    def get_plot(self,on,legend,figout=None):
        """plot embeddings of peptidomic data on train or test"""

        if self.freq_thres !='None':
            fthres='{fthres}%'.format(fthres=round(self.freq_thres*100))
        else:
            fthres=self.freq_thres

        
        
        if on=='train':
            Xtrain_transform=self.clf_pipeline.best_estimator_['umap'].transform(self.Xtrain)
            if legend=='bar':
                plot_bar(Xtrain_transform, 
                    self.ytest,
                    cmapper=self.le,
                    title='Supervised embedding \n $N_{train}=$'+\
                        '{ntrain}'.format(ntrain=len(self.Xtrain))+\
                        '  $F_{thres}=$'+fthres,
                    output1=figout)
            elif legend=='box':
                plot_points(Xtrain_transform,
                    self.le.inverse_transform(self.ytrain),
                    title='Supervised embedding \n $N_{train}=$'+\
                        '{ntrain}'.format(ntrain=len(self.Xtrain))+\
                        '  $F_{thres}=$'+fthres,
                    output1=figout
                    )
            else:
                print ("please specify legend='box' or on='bar'")

        elif on=='test':
            Xtest_transform =self.clf_pipeline.best_estimator_['umap'].transform(self.Xtest)
            if legend=='bar':
                plot_bar(Xtest_transform,self.ytest,cmapper=self.le,
                title='Supervised embedding \n $N_{test}=$'+\
                    '{ntest}'.format(ntest=len(self.Xtest))+\
                    '  $F_{thres}=$'+fthres,
                output1=figout
                )
            elif legend=='box':
                plot_points(Xtest_transform,
                    self.le.inverse_transform(self.ytest),
                    title='Supervised embedding \n $N_{test}=$'+\
                        '{ntest}'.format(ntest=len(self.Xtest))+\
                        '  $F_{thres}=$'+fthres,
                    output1=figout
                    )

            else:
                print ("please specify legend='box' or on='bar'")
                
        else:
            print ("please specify on='test' or on='train'")
