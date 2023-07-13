import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])

    # evidence, labels = load_data("/Users/tasneembenazir/Documents/DYOC/CS50 AI/shopping/shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f, delimiter = ",")
        next(reader)
        evidence_list = []
        label_list = []
        
        for row in reader:
            current_evidence_list = []

            Administrative = int(row[0])
            current_evidence_list.append(Administrative)

            Administrative_Duration = float(row[1])
            current_evidence_list.append(Administrative_Duration)

            Informational = int(row[2])
            current_evidence_list.append(Informational)

            Informational_Duration = float(row[3])
            current_evidence_list.append(Informational_Duration)

            ProductRelated = int(row[4])
            current_evidence_list.append(ProductRelated)

            ProductRelated_Duration = float(row[5])
            current_evidence_list.append(ProductRelated_Duration)

            BounceRates = float(row[6])
            current_evidence_list.append(BounceRates)

            ExitRates = float(row[7])
            current_evidence_list.append(ExitRates)

            PageValues = float(row[8])
            current_evidence_list.append(PageValues)

            SpecialDay = float(row[9])
            current_evidence_list.append(SpecialDay)

            Month = str(row[10])
            if Month in "January":
                Month = 0
            elif Month in "February":
                Month = 1
            elif Month in "March":
                Month = 2
            elif Month in "April":
                Month = 3
            elif Month in "May":
                Month = 4
            elif Month in "June":
                Month = 5
            elif Month in "July":
                Month = 6
            elif Month in "August":
                Month = 7
            elif Month in "September":
                Month = 8
            elif Month in "October":
                Month = 9
            elif Month in "November":
                Month = 10
            else:
                Month = 11
            current_evidence_list.append(Month)

            OperatingSystems = int(row[11])
            current_evidence_list.append(OperatingSystems)

            Browser = int(row[12])
            current_evidence_list.append(Browser)

            Region = int(row[13])
            current_evidence_list.append(Region)

            TrafficType = int(row[14])
            current_evidence_list.append(TrafficType)

            VisitorType = str(row[15])
            if VisitorType == "Returning_Visitor":
                VisitorType = 1
            else:
                VisitorType = 0
            current_evidence_list.append(VisitorType)

            Weekend = str(row[16])
            if Weekend == "TRUE":
                Weekend = 1
            else:
                Weekend = 0
            current_evidence_list.append(Weekend)

            evidence_list.append(current_evidence_list)

            label = str(row[17])
            if label == "TRUE":
                label = 1
            else:
                label = 0
            label_list.append(label)
        
        result = (evidence_list, label_list)
        return result
    

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    total = len(labels)
    TPR_count = 0
    positive_count = 0
    TNR_count = 0
    negative_count = 0

    for idx in range(total):
        actual = labels[idx]
        predict = predictions[idx]
        if actual == 1:
            positive_count += 1
            if actual == predict:
                TPR_count += 1
        if actual == 0:
            negative_count += 1
            if actual == predict:
                TNR_count += 1

    sensitivity = TPR_count/positive_count
    specificity = TNR_count/negative_count
    result = (sensitivity, specificity)
    return result


if __name__ == "__main__":
    main()
