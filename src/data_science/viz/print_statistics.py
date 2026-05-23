
def print_cnf_mtx(mtx):
    print("\t\t       Predicted")
    print("\t\t         N   P")
    print("\t\tTrue  N {}  {}".format(mtx[0,0], mtx[0,1]))
    print("\t\t      P {}  {}".format(mtx[1,0], mtx[1,1]))

def print_cnf_mtx_CT(mtx):
    print("\t\t                Predicted")
    print("\t\t         1   2   3   4   5   6   7")
    print("\t\t      1 {}  {}  {}   {}   {}  {}   {}".format(mtx[0,0], mtx[0,1], mtx[0,2], mtx[0,3], mtx[0,4], mtx[0,5], mtx[0,6]))
    print("\t\t      2 {}  {}  {}  {}   {}  {}   {}".format(mtx[1,0], mtx[1,1], mtx[1,2], mtx[1,3], mtx[1,4], mtx[1,5], mtx[1,6]))
    print("\t\t      3 {}   {}   {}  {}  {}   {}   {}".format(mtx[2,0], mtx[2,1], mtx[2,2], mtx[2,3], mtx[2,4], mtx[2,5], mtx[2,6]))
    print("\t\tTrue  4 {}   {}   {}  {}  {}   {}   {}".format(mtx[3,0], mtx[3,1], mtx[3,2], mtx[3,3], mtx[3,4], mtx[3,5], mtx[3,6]))
    print("\t\t      5 {}   {}   {}   {}   {}  {}   {}".format(mtx[4,0], mtx[4,1], mtx[4,2], mtx[4,3], mtx[4,4], mtx[4,5], mtx[4,6]))
    print("\t\t      6 {}   {}   {}   {}  {} {}  {}".format(mtx[5,0], mtx[5,1], mtx[5,2], mtx[5,3], mtx[5,4], mtx[5,5], mtx[5,6]))
    print("\t\t      7 {}   {}   {}   {}   {}   {}   {}".format(mtx[6,0], mtx[6,1], mtx[6,2], mtx[6,3], mtx[6,4], mtx[6,5], mtx[6,6]))


def print_parameters(classifier, parameters):
    
    if classifier == "Naive Bayes":
        return parameters
    if classifier == "kNN":
        return "distance function = {}; nr neighbors = {}".format(parameters[0], parameters[1])
    if classifier == "Decision Tree":
        return "criteria = {}; max depths = {}; min samples leaf = {}".format(parameters[0], parameters[1], parameters[2])
    if classifier == "Random Forest":
        return "max features = {}; max depths = {}; nr estimators = {}".format(parameters[0], parameters[1], parameters[2])
    if classifier == "Gradient Boost":
        return "max features = {}; max depths = {}; nr estimators = {}; learning rate = {}".format(parameters[0], parameters[1], parameters[2], parameters[3])
    if classifier == "XGBoost":
        return "max depths = {}; nr estimators = {}".format(parameters[0], parameters[1])
    
def print_pre_parameters(pre_parameters):
    balanced = pre_parameters[0]
    normalized = pre_parameters[1]

    if balanced:
        params = "balanced "
    else:
        params = "unbalanced "
    if normalized:
        params += "and normalized"
    else:
        params += "and not normalized"
    return params

def print_analysis(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print("1. Applied preprocessing: {}\n".format(pre_process_params))
    print("2. Classifiers:")
    for i in range(len(reports)):
        print("2.{} {}".format(i+1, reports[i][0]))
        print("\t2.{}.1 Best accuracy".format(i+1))
        best_accuracy = reports[i][1]
        accuracy_parameters = print_parameters(reports[i][0], best_accuracy[0])
        print("\ta) Suggested parameterization: {}".format(accuracy_parameters))
        print("\tb) Confusion matrix: ")
        print_cnf_mtx(best_accuracy[3])
        print("\t2.{}.2 Best specificity".format(i+1))
        best_specificity = reports[i][2]
        specifivity_parameters = print_parameters(reports[i][0], best_specificity[0])
        print("\ta) Suggested parameterization: {}".format(specifivity_parameters))
        print("\tb) Confusion matrix: ")
        print_cnf_mtx(best_specificity[3])
    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format("{0:.2f}".format(report[1][1]))
    print("\t3.1 Accuracy: " + accuracies[:-2])
    specificities = ""
    for report in reports:
        specificities += "{} | ".format("{0:.2f}".format(report[2][2]))
    print("\t3.2 Specificity: " + specificities[:-2])


def print_analysis_CT(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print("1. Applied preprocessing: {}\n".format(pre_process_params))
    print("2. Classifiers:")
    for i in range(len(reports)):
        clf_name = reports[i][0]
        report = reports[i][1]
        parameters = report[0]
        accuracy = report[1]
        cnf_mtx = report[2]
        print("2.{} {}".format(i+1, clf_name))
        accuracy_parameters = print_parameters(clf_name, parameters)
        print("\ta) Suggested parameterization: {}".format(accuracy_parameters))
        print("\tb) Accuracy: {}".format("{0:.2f}".format(accuracy)))
        print("\td) Confusion matrix: ")
        print_cnf_mtx_CT(cnf_mtx)
        
    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format("{0:.2f}".format(report[1][1]))
    print("\t3.1 Accuracy: " + accuracies[:-2])
    

def print_report(reports, pre_parameters):
    pre_process_params = print_pre_parameters(pre_parameters)
    print("1. Applied preprocessing: {}\n".format(pre_process_params))
    print("2. Classifiers:")
    for i in range(len(reports)):
        report = reports[i]
        clf_name = report[0]
        parameters = report[1]
        accuracy = report[2]
        specificity = report[3]
        cnf_mtx = report[4]
        print("2.{} {}".format(i+1, clf_name))
        accuracy_parameters = print_parameters(clf_name, parameters)
        print("\ta) Suggested parameterization: {}".format(accuracy_parameters))
        print("\tb) Accuracy: {}".format("{0:.2f}".format(accuracy)))
        print("\tc) Specificity: {}".format("{0:.2f}".format(specificity)))
        print("\td) Confusion matrix: ")
        print_cnf_mtx(cnf_mtx)
        
    print("3. Comparative performance: NB | kNN | DT | RF | GB | XGB")
    accuracies = ""
    for report in reports:
        accuracies += "{} | ".format("{0:.2f}".format(report[2]))
    print("\t3.1 Accuracy: " + accuracies[:-2])
    specificities = ""
    for report in reports:
        specificities += "{} | ".format("{0:.2f}".format(report[3]))
    print("\t3.2 Specificity: " + specificities[:-2])
        
