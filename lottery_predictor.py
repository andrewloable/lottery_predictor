import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

seq = []
slot1 = []
slot2 = []
slot3 = []
slot4 = []
slot5 = []
slot6 = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            seq.append(int(row[2]))
            slot1.append(int(row[5]))
            slot2.append(int(row[6]))
            slot3.append(int(row[7]))
            slot4.append(int(row[8]))
            slot5.append(int(row[9]))
            slot6.append(int(row[10]))
    return

def predict_numbers(seq, slot1, slot2, slot3, slot4, slot5, slot6, x):
    seq = np.reshape(seq, (len(seq), 1))
    svr_rbf_slot1 = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf_slot2 = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf_slot3 = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf_slot4 = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf_slot5 = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf_slot6 = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr_rbf_slot1.fit(seq, slot1)
    svr_rbf_slot2.fit(seq, slot2)
    svr_rbf_slot3.fit(seq, slot3)
    svr_rbf_slot4.fit(seq, slot4)
    svr_rbf_slot5.fit(seq, slot5)
    svr_rbf_slot6.fit(seq, slot6)

    plt.scatter(seq, slot1, color='black', label='Data')
    plt.scatter(seq, slot2, color='black', label='Data')
    plt.scatter(seq, slot3, color='black', label='Data')
    plt.scatter(seq, slot4, color='black', label='Data')
    plt.scatter(seq, slot5, color='black', label='Data')
    plt.scatter(seq, slot6, color='black', label='Data')

    plt.plot(seq, svr_rbf_slot1.predict(seq), color='red', label="Slot 1")
    plt.plot(seq, svr_rbf_slot2.predict(seq), color='green', label="Slot 2")
    plt.plot(seq, svr_rbf_slot3.predict(seq), color='cyan', label="Slot 3")
    plt.plot(seq, svr_rbf_slot4.predict(seq), color='yellow', label="Slot 4")
    plt.plot(seq, svr_rbf_slot5.predict(seq), color='blue', label="Slot 5")
    plt.plot(seq, svr_rbf_slot6.predict(seq), color='pink', label="Slot 6")
    plt.xlabel('Date')
    plt.ylabel('Numbers')
    plt.title('Lottery Predictions - Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf_slot1.predict(x)[0], svr_rbf_slot2.predict(x)[0], svr_rbf_slot3.predict(x)[0], svr_rbf_slot4.predict(x)[0], svr_rbf_slot5.predict(x)[0], svr_rbf_slot6.predict(x)[0]

get_data('658_results_draw_sort.csv')

predicted = predict_numbers(seq, slot1, slot2, slot3, slot4, slot5, slot6, 265)

print "Next Prediction Sequence" 
print (predicted)