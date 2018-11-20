import numpy as np
from sklearn.model_selection import train_test_split
from NCH import NCHClassifier
from sklearn.metrics import accuracy_score

def load_iris():
    samples = []
    labels = []
    with open('data/iris.data.txt', 'r') as f:
        lines = f.read()
        data = lines.split('\n')
        for item in data:
            x = item.split(',')
            samples.append(x[0:-1])
            labels.append(x[-1])
    label_set = list(set(labels))
    y = [label_set.index(i) for i in labels]
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return samples, y

def load_aca():
    samples = []
    labels = []
    with open('data/aca.txt', 'r') as f:
        lines = f.read()
        data = lines.split('\n')
        for item in data:
             x = item.split(' ')
             if len(x) == 15:
                 samples.append(x[0:-1])
                 labels.append(x[-1])
    samples = np.asarray(samples, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return samples, labels

if __name__ == "__main__":
    samples, labels = load_aca()
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)    
    y_pred = []
    n_classes = len(set(labels))
    for test_point in X_test:
        distance = []
        for label in range(n_classes):
            points = X_train[np.where(y_train == label)]
            test_point = test_point.reshape(1, test_point.size)
            nch_clf = NCHClassifier(points, test_point, 1)
            distance.append(nch_clf.solve())
        y_pred.append(np.argmax(distance))
    print(accuracy_score(y_test, y_pred))
            
    
