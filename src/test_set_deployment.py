import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def test_set_deployment():
    comparison_df = pd.read_csv('../results/tables/comparison_df.csv', index_col=0)
    svc_C = comparison_df.loc['svc', 'param_model__C']
    svc_gamma = comparison_df.loc['svc', 'param_model__gamma']
    svc_class_weight = None if comparison_df.loc['svc', 'param_model__class_weight'] == 'No Class Weight' else 'balanced'
    
    X_test = pd.read_csv('../results/tables/X_test.csv')
    y_test = (pd.read_csv('../results/tables/y_test.csv')).iloc[:, 0]
    X_train = pd.read_csv('../results/tables/X_train.csv')
    y_train = (pd.read_csv('../results/tables/y_train.csv')).iloc[:, 0]
    
    best_pipe = make_pipeline(StandardScaler(), SVC(C=svc_C, gamma=svc_gamma, class_weight=svc_class_weight))
    best_pipe.fit(X_train, y_train)
    
    performance = pd.DataFrame({'test_set_score': [best_pipe.score(X_test, y_test)]})
    
    performance.to_csv(('../results/tables/test_set_score.csv'), index=True)

if __name__ == '__main__':
    test_set_deployment()

