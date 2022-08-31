from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def create_collection_classifier():

    collection = []

    collection.append(('DT', DecisionTreeClassifier(max_depth=3)))
    collection.append(('gb', GaussianNB()))

    k_neigh = [1, 3, 5, 10, 15, 20, 30]

    for j in k_neigh:
        knn_k = KNeighborsClassifier(n_neighbors=j)
        collection.append(('knn[' + str(j) + ']', knn_k))

    return collection

def Voting_Classifier(X_train, Y_train, X_test):

    collection_VC = create_collection_classifier()

    vot_hard = VotingClassifier(estimators=collection_VC, voting='soft')
    vot_hard.fit(X_train, Y_train)
    pred = vot_hard.predict_proba(X_test)

    return pred

def Stacking_Classifier (X_train, Y_train, X_test, k):

    collection_SC = create_collection_classifier()

    sc = StackingClassifier(estimators=collection_SC)
    sc.fit(X_train, Y_train)
    sc_pred = sc.predict_proba(X_test)

    return sc_pred








