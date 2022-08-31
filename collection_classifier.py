from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def Voting_Classifier(X_train, Y_train, X_test, k):

    collection_VC = []

    collection_VC.append(('DT', DecisionTreeClassifier(max_depth=3)))
    collection_VC.append(('gb', GaussianNB()))

    knn_k = KNeighborsClassifier(n_neighbors=k)
    collection_VC.append(('knn', knn_k))

    vot_hard = VotingClassifier(estimators=collection_VC, voting='soft')
    vot_hard.fit(X_train, Y_train)
    pred = vot_hard.predict_proba(X_test)

    return pred

def Stacking_Classifier (X_train, Y_train, X_test, k):

    collection_SC = []

    collection_SC.append(('DT', DecisionTreeClassifier(max_depth=3)))
    collection_SC.append(('gb', GaussianNB()))

    knn_k = KNeighborsClassifier(n_neighbors=k)
    collection_SC.append(('knn', knn_k))

    sc = StackingClassifier(estimators=collection_SC)
    sc.fit(X_train, Y_train)
    sc_pred = sc.predict_proba(X_test)

    return sc_pred








