import json
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve

# from sklearn.metrics import accuracy_score

json1_data = None
json2_data = None


# import compute_feature

def read_dataprofilers(file1, file2):
    json1_file = open(file1).read()
    json2_file = open(file2).read()
    global json1_data
    global json2_data
    json1_data = json.loads(json1_file)
    json2_data = json.loads(json2_file)


def isPairOfFieldNumeric(field1, field2):
    if "numeric_data_stats" in json1_data[field1]["length"] and "numeric_data_stats" in json2_data[field2]["length"]:
        return True
    return False


def read_trainingdata(matchfile, nonmatchfile):
    with open(matchfile, "rb") as f:
        data1 = f.read().split('\n')

    with open(nonmatchfile, "rb") as f:
        data2 = f.read().split('\n')

    data = data1 + data2
    random.shuffle(data)
    d1 = []
    d2 = []
    for d in data:
        split_data = d.split("\t")
        if isPairOfFieldNumeric(split_data[0], split_data[1]):
            d1.append(d)
        else:
            d2.append(d)

    return d2, d1
    # return data


def feature_lencommonpunc(field1, field2):
    set1 = set()
    set2 = set()
    temp_set1 = json1_data[field1]['frequent-entries']['most_common_punctuation']
    temp_set2 = json2_data[field2]['frequent-entries']['most_common_punctuation']

    for k, v in temp_set1.iteritems():
        if '\\' in k:
            continue
        set1.add(k)
    for k, v in temp_set2.iteritems():
        if '\\' in k:
            continue
        set2.add(k)
    intersect_arr = set1.intersection(set2)
    return len(intersect_arr)


def feature_lencommonvalues(field1, field2):
    set1 = set()
    set2 = set()

    temp_set1 = json1_data[field1]['frequent-entries']['most_common_values']
    temp_set2 = json2_data[field2]['frequent-entries']['most_common_values']

    for k, v in temp_set1.iteritems():
        set1.add(k)
    for k, v in temp_set2.iteritems():
        set2.add(k)

    intersect_arr = set1.intersection(set2)
    return len(intersect_arr)


def feature_lencommontokens(field1, field2):
    set1 = set()
    set2 = set()

    temp_set1 = json1_data[field1]['frequent-entries']['most_common_tokens']
    temp_set2 = json2_data[field2]['frequent-entries']['most_common_tokens']

    for k, v in temp_set1.iteritems():
        set1.add(k)
    for k, v in temp_set2.iteritems():
        set2.add(k)

    intersect_arr = set1.intersection(set2)
    return len(intersect_arr)


def feature_diff_token_avg(field1, field2):
    avg1 = json1_data[field1]['length']['token']['average']
    avg2 = json2_data[field2]['length']['token']['average']
    return avg1 - avg2


def feature_diff_token_std(field1, field2):
    std1 = json1_data[field1]['length']['token']['standard-deviation']
    std2 = json2_data[field2]['length']['token']['standard-deviation']
    return std1 - std2


def feature_diff_char_avg(field1, field2):
    avg1 = json1_data[field1]['length']['character']['average']
    avg2 = json2_data[field2]['length']['character']['average']
    return avg1 - avg2


def feature_diff_char_std(field1, field2):
    std1 = json1_data[field1]['length']['character']['standard-deviation']
    std2 = json2_data[field2]['length']['character']['standard-deviation']
    return std1 - std2


def feature_lencommondatatypes(field1, field2):
    datatype1 = json1_data[field1]['type']
    datatype2 = json2_data[field2]['type']
    datatype_intersect = set(datatype1).intersection(datatype2)
    return len(datatype_intersect)


def feature_diff_numeric_stat_std(field1, field2):
    std1 = json1_data[field1]['length']['numeric_data_stats']['standard-deviation']
    std2 = json2_data[field2]['length']['numeric_data_stats']['standard-deviation']
    return std1 - std2


def feature_diff_numeric_stat_min(field1, field2):
    min1 = json1_data[field1]['length']['numeric_data_stats']['min']
    min2 = json2_data[field2]['length']['numeric_data_stats']['min']
    return min1 - min2


def feature_diff_numeric_stat_max(field1, field2):
    max1 = json1_data[field1]['length']['numeric_data_stats']['max']
    max2 = json2_data[field2]['length']['numeric_data_stats']['max']
    return max1 - max2


def feature_diff_numeric_stat_avg(field1, field2):
    # print field1 + " " + field2
    avg1 = json1_data[field1]['length']['numeric_data_stats']['average']
    avg2 = json2_data[field2]['length']['numeric_data_stats']['average']
    return avg1 - avg2


def feature_diff_numeric_stat_mode(field1, field2):
    mode1 = json1_data[field1]['length']['numeric_data_stats']['mode']
    mode2 = json2_data[field2]['length']['numeric_data_stats']['mode']
    return mode1 - mode2


def get_features(field1, field2, nummeric_Feature=False):
    feature = []
    feature.append(feature_lencommonpunc(field1, field2))
    # feature.append(feature_lencommonvalues(field1,field2))
    feature.append(feature_lencommontokens(field1, field2))

    feature.append(feature_lencommondatatypes(field1, field2))
    feature.append(feature_diff_char_avg(field1, field2))
    # feature.append(feature_diff_char_std(field1,field2))
    feature.append(feature_diff_token_avg(field1, field2))
    # feature.append(feature_diff_token_std(field1,field2))

    if nummeric_Feature:
        feature.append(feature_diff_numeric_stat_avg(field1, field2))
        feature.append(feature_diff_numeric_stat_std(field1, field2))
        feature.append(feature_diff_numeric_stat_min(field1, field2))
        feature.append(feature_diff_numeric_stat_max(field1, field2))

    return feature


def feature_generation(data, numericFeatures=False):
    train_data = data
    test_data = data
    feature_arr = []
    output_arr = []
    label_field = []
    for d in train_data:
        arr = d.split("\t")

        field1 = arr[0]
        field2 = arr[1]
        match = arr[2]

        feature = get_features(field1, field2, numericFeatures)
        label_field.append(field1 + ":" + field2)
        feature_arr.append(feature)
        if 'Not Same pair' in match:
            output_arr.append(0)
        else:
            output_arr.append(1)
    return feature_arr, output_arr


def plot_graph(feature_arr, output_arr, xlabel='features', ylabel='label match', title='classfication dbpedia',
                  filename='test.png'):
    plt.plot(feature_arr, output_arr, 'ro')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def accuracy(predicted, testinglabels):
    length = len(predicted)

    count = 0
    for i in range(0, len(predicted)):
        # print str(predicted[i]) + ":" + str(testinglabels[i])
        if predicted[i] == testinglabels[i]:
            count += 1
    res = float(count) / float(length) * 100.0
    print "Accuracy is : " + str(res)

    y_true = np.array(testinglabels)
    y_pred = np.array(predicted)
    print "precision and recall"
    print precision_recall_fscore_support(y_true, y_pred, average='macro')
    '''
    print "curve"
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    print precision
    print recall
    print thresholds
    '''


def accuracy1(predicted, testinglabels, title=""):
    '''
    y_true = []
    for l in testinglabels:
        if l == 0:
            y_true.append([0,1])
        else:
            y_true.append([1,0])
    '''
    y_pred1 = []
    y_pred2 = []
    for i in predicted:
        y_pred1.append(i[0])
        y_pred2.append(i[1])

    y_true = np.array(testinglabels)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    '''
    print "precision and recall"
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred1)
    plot_graph(recall,precision,"recall","precision","recall-precision","recall-precision_"+title+"_pos.png")
    print precision
    print recall
    print thresholds
    '''
    print "precision and recall curve for " + title
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred2)
    plot_graph(recall, precision, "recall", "precision", "recall-precision",
                  "recall-precision_mod_" + title + "_pos.png")
    '''
    print precision
    print recall
    print thresholds
    '''


def generate_classifier(training_feature, trainging_label):
    # feature_label(feature_arr,output_arr)
    clf = SVC(probability=True)
    clf.fit(training_feature, trainging_label)

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    return clf


def precision_recall_k(filename, clf, clf_num):
    with open(filename, "rb") as f:
        data1 = f.read().split('\n')
    d1 = []
    d2 = []

    precision_k = []
    recall_k = []
    match_point_k = []
    #count = len(data1) + 1
    for d in data1:
        #count -= 1
        #if count<0:
        #    print "I am breaking here"
        #    break
        split_data = d.split("\t")
        if isPairOfFieldNumeric(split_data[0], split_data[1]):
            d1.append(split_data[0])
        else:
            d2.append(split_data[0])

    matrix = []
    for f1 in range(0,len(d1)):
        matrix.append([])
        for f2 in range(0,len(d1)):
            # field1, field2, probability, precision, recall
            tup = [d1[f1],d1[f2],0.0,0.0,0.0]
            matrix[f1].append(tup)
    for m in matrix:
        data = []
        #feature generation for alphnum
        for l in m:
            data.append(l[0] + "\t" + l[1] + "\t" +"Not Same pair")
        feature_arr, output_arr = feature_generation(data, False)


        predicted = clf.predict_proba(feature_arr)
        #print predicted
        for i in range(0,len(predicted)):
            #print m[i][2]
            #print predicted[i][1]
            m[i][2] = predicted[i][1]
        m.sort(key=lambda x: x[2],reverse=True)
        match_index = -1
        for i in range(len(m)):
            if m[i][0] == m[i][1]:
                match_index = i
                break
        #match_index += 1
        match_point_k.append(match_index+1)
        precision_k.append(1.0/(match_index+1))
        recall_k.append(1)
        for i in range(0,len(m)):
            if i >= match_index:
                m[i][3] = 1.0/(match_index+1)
            else:
                m[i][3] = 0.0
            if i>=match_index:
                m[i][4] = 1
            else:
                m[i][4] = 0

    print "printing matrices"
    for i in range(len(match_point_k)):
        print str(match_point_k[i]) + " " +  str(precision_k[i])
    plot_graph(precision_k,match_point_k,"precision","k","precision at match point k", "precision_k_test.png")
    #print matrix

def predict(dataProfiler1, dataProfiler2, labelledData1, labelledData2, className, clf, clf_num):
    read_dataprofilers(dataProfiler1, dataProfiler2)
    #precision_recall_k("pairs1.txt")
    precision_recall_k(labelledData1, clf, clf_num)
    data, data2 = read_trainingdata(labelledData1, labelledData2)
    feature_arr, output_arr = feature_generation(data, False)
    predicted = clf.predict_proba(feature_arr)

    feature_arr1, output_arr1 = feature_generation(data2, True)
    predicted_num = clf_num.predict_proba(feature_arr1)
    #print(len(feature_arr))
    #print len(predicted)

    #print "predicted is "
    #for p in predicted:
    #    print p

    # merged_predicted_values = predicted + predicted_num
    # merged_output = output_arr + output_arr1
    # accuracy1(merged_predicted_values,merged_output,className)
    accuracy1(predicted, output_arr, className)
    accuracy1(predicted_num, output_arr1, className + " numeric")


if __name__ == '__main__':
    read_dataprofilers('data_stats1.json', 'data_stats2.json')
    data, data1 = read_trainingdata("pairs.txt", "non_pairs.txt")

    # read_dataprofilers('university_profile.json', 'university_profile1.json')
    # data, data1 = read_trainingdata("university_pairs.txt", "university_non_pairs.txt")

    # read_dataprofilers('Organisation_profile5k.json','Organisation_profile5k1.json')
    # data, data1 = read_trainingdata("pairs1.txt", "non_pairs1.txt")

    print "len of alphanum and num"
    print len(data)
    print len(data1)

    k1 = int(len(data) / 2)
    _k1 = len(data)
    feature_arr, output_arr = feature_generation(data, False)

    # training data
    training_feature = feature_arr[:_k1]
    trainging_label = output_arr[:_k1]

    # testing data
    testingfeature = feature_arr[k1:]
    testinglabels = output_arr[k1:]

    k2 = int(len(data1))
    _k2 = int(len(data1) / 2)

    feature_arr1, output_arr1 = feature_generation(data1, True)

    # training numeric data
    training_feature1 = feature_arr1[:k2]
    trainging_label1 = output_arr1[:k2]

    # testing numeric data
    testingfeature1 = feature_arr1[_k2:]
    testinglabels1 = output_arr1[_k2:]

    # feature_label(feature_arr,output_arr)
    clf = generate_classifier(training_feature, trainging_label)
    clf_numeric = generate_classifier(training_feature1, trainging_label1)

    # predict Person
    # predicted = clf.predict_proba(testingfeature)
    # print predicted
    # accuracy1(predicted,testinglabels,"Person")

    print "-------------------------"
    # predict Organisation
    predict('Organisation_profile5k.json', 'Organisation_profile5k1.json', "pairs1.txt", "non_pairs1.txt",
            "Organisation", clf, clf_numeric)
    # predict('university_profile.json','university_profile1.json','university_pairs.txt','university_non_pairs.txt',"University",clf,clf_numeric)
    # predict('data_stats1.json', 'data_stats2.json',"pairs.txt", "non_pairs.txt","Person",clf,clf_numeric)

    '''
    read_dataprofilers('Organisation_profile5k.json','Organisation_profile5k1.json')
    data = read_trainingdata("pairs1.txt", "non_pairs1.txt")
    feature_arr, output_arr = feature_generation(data)
    predicted = clf.predict_proba(feature_arr)
    accuracy1(predicted,output_arr,"Organisation")
    '''
