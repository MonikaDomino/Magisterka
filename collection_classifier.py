from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def create_collection_classifier():

    collection = []

    collection.append(('DT', DecisionTreeClassifier(max_depth=3)))
    collection.append(('gb', GaussianNB()))
    collection.append(('lr', LogisticRegression(max_iter=3000)))

    k_neigh = [1, 3, 5, 10, 15, 20, 30]

    for j in k_neigh:
        knn_k = KNeighborsClassifier(n_neighbors=j)
        collection.append(('knn[' + str(j) + ']', knn_k))

    return collection

def voting_classifier_hard (X_train, Y_train, X_test):

    collection_VCH = create_collection_classifier()

    vot_hard = VotingClassifier(estimators=collection_VCH, voting='soft')
    vot_hard.fit(X_train, Y_train)
    pred = vot_hard.predict_proba(X_test)[:,1]
    return pred

def voting_classifier_soft (X_train, Y_train, X_test):

    collection_VCS = create_collection_classifier()

    vot_soft = VotingClassifier(estimators=collection_VCS, voting='soft')
    vot_soft.fit(X_train, Y_train)
    predS = vot_soft.predict_proba(X_test)[:,1]

    return predS


def stacking_classifier (X_train, Y_train, X_test):

    collection_SC = create_collection_classifier()

    sc = StackingClassifier(estimators=collection_SC)
    sc.fit(X_train, Y_train)
    sc_pred = sc.predict_proba(X_test)[:,1]

    return sc_pred








