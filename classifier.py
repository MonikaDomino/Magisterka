from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from AUC import  AUC_intervals
from sklearn.linear_model import LogisticRegression
import check_dataset
from collection_classifier import voting_classifier_hard, stacking_classifier, voting_classifier_soft


data = check_dataset.readDataset_onlineShopeersIntention()

X = data.drop(['Revenue'], axis=1)
Y = data['Revenue']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

scaler= StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# classifiers - KNN, Decision Tree, Gausian Native Bayes, LogisticRegression

def KNN_clasifier_predict(k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, Y_train)
    knn_score = knn.predict_proba(X_test)[:,1]
    return knn_score

def LogisticRegression_predict():
    logic = LogisticRegression(max_iter=3000)
    logic.fit(X_train, Y_train)
    logistic_score = logic.predict_proba(X_test)[:,1]
    return logistic_score

def DecisionTree_classifier_predict():
    tre = DecisionTreeClassifier(max_depth=5)
    tre.fit(X_train, Y_train)
    tree_score = tre.predict_proba(X_test)[:,1]
    return tree_score


def GaussianNativeBayes_predict():
    gaussian = GaussianNB()
    gb = gaussian.fit(X_train, Y_train)
    g_score = gb.predict_proba(X_test)[:,1]
    return g_score


def auc_classifier_knn():
    k_neigh = [1, 3, 5, 10, 15, 20, 25, 30]

    for j in k_neigh:
        knn_k = KNN_clasifier_predict(j)
        print("AUC KNN (k =", j, '):', AUC_intervals(Y_test, knn_k))


tree = DecisionTree_classifier_predict()
gaussian = GaussianNativeBayes_predict()
log = LogisticRegression_predict()

votCH = voting_classifier_hard(X_train, Y_train, X_test)
votS = voting_classifier_soft(X_train, Y_train, X_test)

sC = stacking_classifier(X_train, Y_train, X_test)

auc_classifier_knn()

print()
print("AUC Decision Tree: ", AUC_intervals(Y_test, tree))
print("AUC Gausian Native Bayes: ", AUC_intervals(Y_test, tree))
print("AUC Logistic Regreesion: ", AUC_intervals(Y_test, log))
print()
print("AUC Voting Classier hard: ", AUC_intervals(Y_test, votCH))
print("AUC Voting Classier soft: ", AUC_intervals(Y_test,votS))
print("AUC Stacking Classifier: ", AUC_intervals(Y_test,sC))
