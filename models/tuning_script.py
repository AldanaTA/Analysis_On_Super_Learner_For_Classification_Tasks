from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


def grid_search(model,parameters,X,Y):
    scorers = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score,average="micro"),
    'recall': make_scorer(recall_score,average="micro"),
    'f1': make_scorer(f1_score,average="micro")
}
    grid_obj = GridSearchCV(model, parameters, scoring=scorers, cv=5,refit='accuracy')
    grid_obj.fit(X,Y)
    print("Best Hyper paramaters: ",grid_obj.best_params_)
    print("Best Scores: {0:.2%}".format(grid_obj.best_score_))
    return grid_obj.best_params_
