'''
Created on Aug 7, 2020

@author: jpereira
'''
import os
import csv
import re
import random
import math
import matplotlib.pyplot as plt

import numpy as np
from string_constants import folder_logs

from Logger import logging

logger = logging.getLogger(__name__)

#folder_logs = "/home/jpereira/MEGA/Acronym/results/vldb/gcloud_logs/merge_logs/"
folder_logs = "/home/jpereira/MEGA/Acronym/results/vldb/uva_logs/"

def save_to_file(datasetname, technique1, technique2, score_diff, p_value):
    file_path = folder_logs + "statistical_tests.csv"
    row_to_write={"Dataset":datasetname,
                  "technique1": technique1,
                  "technique2": technique2,
                  "score_diference": score_diff,
                  "p-value": p_value}
    
    with open(file_path, 'a') as file:
        w = csv.DictWriter(file, row_to_write.keys())

        if file.tell() == 0:
            w.writeheader()

        w.writerow(row_to_write)

def swamp_scores(scores_1, scores_2):
    scores_1 = scores_1.copy()
    scores_2 = scores_2.copy()
    
    for i in range(len(scores_1)):
        if random.random() >= 0.5:
            #swamp values
            aux = scores_1[i]
            scores_1[i] = scores_2[i]
            scores_2[i] = aux
    
    return scores_1, scores_2

def non_parametric_test(scores_1, scores_2, 
                              func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0):
    
    random.seed(seed)
    score_diff=func(scores_1, scores_2)
    successes = 0
    for i in range(num_rounds):
        swamped_s1, swamped_s2 = swamp_scores(scores_1, scores_2)
        swamped_diff = func(swamped_s1, swamped_s2)
        
        if swamped_diff >= score_diff:
            successes +=1
    
    p_value = successes / num_rounds
    return p_value

def process_out_expansion_results_file(datasetname, out_expander):
    filename = folder_logs + "quality_results_"+ datasetname+ "_" + out_expander + ".csv" 
    
    fold = None
    docIds = []
    total_expansions = 0
    scores_per_article_list = []
    
    total_sucesses = 0
    
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        
        current_doc_id = None
        current_doc_sucesses = 0
        for row in reader:

            if not fold:
                fold = row['fold']
            elif fold != row['fold']:
                logger.warning("Different folds found in same results file: %s and %s",fold, row['fold'])
                
            if current_doc_id != row['doc id']:
                if current_doc_id:
                    docIds.append(current_doc_id)
                    scores_per_article_list.append(current_doc_sucesses)
                
                current_doc_sucesses = 0
                current_doc_id = row['doc id']
                
            if row['success'] == "True":
                current_doc_sucesses +=1
                total_sucesses+=1
            elif row['success'] != "False":
                logger.critical("Unkown success value: %s", row['success'])

            total_expansions += 1
        
        docIds.append(current_doc_id)
        scores_per_article_list.append(current_doc_sucesses)
        
    return fold, docIds, total_expansions, np.array(scores_per_article_list)
            
def compute_accuracy(scores_per_article, total_expansions):
    return np.sum(scores_per_article) / total_expansions

def compute_paired_test(datasetname, out_expander1, out_expander2):
    try:
        fold1, docIds1, total_expansions1, scores_per_article1 = process_out_expansion_results_file(datasetname, out_expander1)
        
        fold2, docIds2, total_expansions2, scores_per_article2 = process_out_expansion_results_file(datasetname, out_expander2)
    except Exception:
        logger.exception("A test failed for " + out_expander1 + " vs " + out_expander2)
        return None, None
        
    if (fold1, docIds1, total_expansions1) != (fold2, docIds2, total_expansions2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
        return None, None

    total_expansions = total_expansions1

    func = lambda x, y: compute_accuracy(x, total_expansions) - compute_accuracy(y, total_expansions)
    logger.critical("Accuracy1: %f", compute_accuracy(scores_per_article1, total_expansions))
    logger.critical("Accuracy2: %f", compute_accuracy(scores_per_article2, total_expansions))
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical("Accuracy difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    
    p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                               func=func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0)
    logger.critical("P-Value: %f", p_value)
    
    save_to_file(datasetname, out_expander1, out_expander2, score_diff, p_value)
    
    return score_diff, p_value


def compute_tests_for_technique(datasetname, technique, only_exec = False, end_to_end = False):
    if not only_exec:
        for file in os.listdir(folder_logs):
            m = re.match('^quality_results_'+datasetname+'([.:=_\-\\w]+).csv$', file)
            if m:
                technique_2 = m.group(1)
                if not end_to_end:
                    compute_paired_test(datasetname, technique, technique_2)
                else:
                    compute_paired_test_f1(datasetname, technique, technique_2)
                    
    for file in os.listdir(folder_logs):
        m_exec = re.match('^exec_time_results_'+datasetname+'([.:=_\-\\w]+).csv$', file)
        if m_exec:
            technique_2 = m_exec.group(1)
            compute_paired_test_exec_times(datasetname, technique, technique_2)


def process_end_to_end_expansion_results_file(datasetname, out_expander):
    #filename = folder_logs + "results_extraction_"+ datasetname+ "_" + out_expander + ".csv" 
    filename = folder_logs + "quality_results_"+ datasetname + out_expander + ".csv" 
    
    #fold = None
    docIds = []
    total_expansions = 0
    scores_per_article_list = []
    
    total_sucesses = 0
    
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        
        current_doc_id = None
        current_doc_tp = 0
        current_doc_fp = 0
        current_doc_fn = 0
        for row in reader:

            #if not fold:
            #    fold = row['fold']
            #elif fold != row['fold']:
            #    logger.warning("Different folds found in same results file: %s and %s",fold, row['fold'])
                
            if current_doc_id != row['doc id']:
                if current_doc_id:
                    docIds.append(current_doc_id)
                    scores_per_article_list.append((current_doc_tp, current_doc_fp, current_doc_fn))
                
                current_doc_tp = 0
                current_doc_fp = 0
                current_doc_fn = 0
                current_doc_id = row['doc id']
                
            if row['success'] == "True":
                current_doc_tp +=1
                total_sucesses+=1
                total_expansions += 1
            elif row['success'] != "False":
                logger.critical("Unkown success value: %s", row['success'])
            else:
                if row['predicted_expansion'] == "":
                    current_doc_fn += 1
                    total_expansions += 1
                else:
                    current_doc_fp += 1
                    if row['actual_expansion'] != "":
                        total_expansions += 1
        
        docIds.append(current_doc_id)
        scores_per_article_list.append((current_doc_tp, current_doc_fp, current_doc_fn))
        
        scores_per_article_list = [x for _,x in sorted(zip(docIds,scores_per_article_list))]
    return sorted(docIds), total_expansions, np.array(scores_per_article_list)

def compute_precision(scores_per_article):
    tp, fp, fn = np.sum(scores_per_article, axis=0)
    return tp/ (tp + fp)

def compute_recall(scores_per_article):
    tp, fp, fn = np.sum(scores_per_article, axis=0)
    return tp/ (tp + fn)

def compute_f1(scores_per_article):
    tp, fp, fn = np.sum(scores_per_article, axis=0)
    return tp / (tp + 0.5 * (fp + fn))

def compute_paired_test_f1(datasetname, out_expander1, out_expander2):
    docIds1, total_expansions1, scores_per_article1 = process_end_to_end_expansion_results_file(datasetname, out_expander1)
    
    docIds2, total_expansions2, scores_per_article2 = process_end_to_end_expansion_results_file(datasetname, out_expander2)
    
    if (docIds1, total_expansions1) != (docIds2, total_expansions2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
    #    return None, None

    func = lambda x, y: compute_recall(x) - compute_recall(y)
    
    logger.critical("Precision: %f", compute_precision(scores_per_article1))
    logger.critical("Precision: %f", compute_precision(scores_per_article2))
    
    logger.critical("Recall: %f", compute_recall(scores_per_article1))
    logger.critical("Recall: %f", compute_recall(scores_per_article2))
    
    logger.critical("F1: %f", compute_f1(scores_per_article1))
    logger.critical("F1: %f", compute_f1(scores_per_article2))
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical("F1 difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    try:
        p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                                   func=func,
                                   method='approximate',
                                   num_rounds=1000,
                                   seed=0)
        logger.critical("P-Value: %f", p_value)
        save_to_file(datasetname, out_expander1, out_expander2, score_diff, p_value)
        return score_diff, p_value
    
    except Exception:
        logger.exception("Failed to compute P-value for results files: %s %s", out_expander1, out_expander2)
        return None, None



def process_execution_times_results_file(datasetname, out_expander):
    filename = folder_logs + "exec_time_results_"+ datasetname+ "" + out_expander + ".csv" 
    
    fold = None
    docIds = []
    exec_times_per_article_list = []
        
    with open(filename, 'r' ) as f:
        reader = csv.DictReader(f)
        try:
            for row in reader:
                docIds.append(row['doc id'])
                exec_times_per_article_list.append(float(row['Execution Times']))
                if not fold:
                    fold = row['fold']
                elif fold != row['fold']:
                    logger.warning("Different folds found in same results file: %s and %s",fold, row['fold'])
        except Exception:
            logger.exception("Error reading file: " + filename)
        exec_times_per_article_list = [x for _,x in sorted(zip(docIds,exec_times_per_article_list))]
    return sorted(docIds), len(docIds), np.array(exec_times_per_article_list)

def compute_exec_time_avg(exec_times_per_article_list):
    avg = np.mean(exec_times_per_article_list)
    return avg

def compute_paired_test_exec_times(datasetname, out_expander1, out_expander2):
    docIds1, total_expansions1, scores_per_article1 = process_execution_times_results_file(datasetname, out_expander1)
    
    docIds2, total_expansions2, scores_per_article2 = process_execution_times_results_file(datasetname, out_expander2)
    
    if (docIds1, total_expansions1) != (docIds2, total_expansions2):
        logger.error("Incompatible results files: %s %s", out_expander1, out_expander2)
        return None, None

    func = lambda x, y: compute_exec_time_avg(x) - compute_exec_time_avg(y)
    logger.critical("Execution Times: %f", compute_exec_time_avg(scores_per_article1))
    logger.critical("Execution Times: %f", compute_exec_time_avg(scores_per_article2))
     
    score_diff=func(scores_per_article1, scores_per_article2)
    logger.critical("Execution Times difference: %f", score_diff)
    
    if score_diff < 0:
        aux = scores_per_article1
        scores_per_article1 = scores_per_article2
        scores_per_article2 = aux
    
    p_value = non_parametric_test(scores_per_article1, scores_per_article2,
                               func=func,
                               method='approximate',
                               num_rounds=1000,
                               seed=0)
    logger.critical("P-Value: %f", p_value)
    if p_value > 0.005:
        logger.critical("----------------------------------------------------not significant for+ " + out_expander2)
    
    save_to_file("exec_time_" + datasetname, out_expander1, out_expander2, score_diff, p_value)
    
    return score_diff, p_value


"""
    Confidence intervals code
"""
def bootstrap(x):
        samp_x = []
        for i in range(len(x)):
                samp_x.append(random.choice(x))
        return samp_x
    
def compute_confidence_intervals(datasetname, out_expander):
    conf_interval = 0.9
    num_resamples = 10000   # number of times we will resample from our original samples

    fold, docIds, total_expansions, scores_per_article = process_out_expansion_results_file(datasetname, out_expander)
    accuracy_list = []
    n_articles = len(scores_per_article)
    accuracy = compute_accuracy(scores_per_article, total_expansions)
    print("Accuracy: " + str(accuracy))
    for i in range(num_resamples):
        #random take n_articles with repetition from scores_per_article1
        new_scores = bootstrap(scores_per_article)
        acc = compute_accuracy(new_scores, total_expansions)
        accuracy_list.append(acc)
        
    accuracy_list.sort()

    # standard confidence interval computations
    tails = (1 - conf_interval) / 2
    
    # in case our lower and upper bounds are not integers,
    # we decrease the range (the values we include in our interval),
    # so that we can keep the same level of confidence
    lower_bound = int(math.ceil(num_resamples * tails))
    upper_bound = int(math.floor(num_resamples * (1 - tails)))
    
    print("Lower bound: " + str(accuracy_list[lower_bound]))
    print("Upper bound: " + str(accuracy_list[upper_bound]))
    return  accuracy, (accuracy_list[upper_bound] - accuracy, accuracy - accuracy_list[lower_bound])
        
def plot_histogram_confidences_interval(expanders_names, accuracies, intervals):
    x_pos = np.arange(len(expanders_names))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, accuracies, yerr= np.asarray(intervals).T, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Out-Expansion Accuracy (%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(expanders_names)
    ax.set_title('Out-Expansion Accuracy Coffidence Intervals for 90% cofidence.')
    ax.yaxis.grid(True)
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()
    
if __name__ == '__main__':
    #compute_paired_test("ScienceWISE", "SVM_Doc2Vec_6_l2_0.1", "DualEncoder_1_6_2")
    
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Concat1_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Concat1_5_l2_0.01")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Concat1_5_l2_0.01")
    """
    
    """
    compute_tests_for_technique("ScienceWISE", "ContextVector_1")
    compute_tests_for_technique("MSHCorpusSOA", "ContextVector_1")
    compute_tests_for_technique("CSWikipedia_res-dup","ContextVector_1")
    """
    
    #compute_paired_test("ScienceWISE","SVM_Concat1_6_l2_0.1", "ContextVector_1")
    #compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","with_links_Orig_SH_SVM_Concat1_5_l2_0.01")
    #compute_paired_test_exec_times("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","with_links_Orig_SH_SVM_Concat1_5_l2_0.01")
    """
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_ContextVector_1","Orig_SH_SVM_Concat1_5_l2_0.01")
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_ContextVector_1","Orig_SH_SVM_Doc2Vec_5_l2_0.01")
    compute_paired_test_f1("FullWikipedia_UsersWikipedia","Orig_SH_SVM_Concat1_5_l2_0.01","Orig_SH_SVM_Doc2Vec_5_l2_0.01")
    """
    
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Concat2_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Concat2_6_l2_0.1")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Concat2_5_l2_0.01")
    """
    """
    compute_tests_for_technique("ScienceWISE", "SVM_Doc2Vec_6_l2_0.1")
    compute_tests_for_technique("MSHCorpusSOA", "SVM_Doc2Vec_6_l2_0.1")
    compute_tests_for_technique("CSWikipedia_res-dup","SVM_Doc2Vec_5_l2_0.01")
    """
    
    #accuracy, interval = compute_confidence_intervals("ScienceWISE", "cossim_document_context_vector")
    #accuracy2, interval2 = compute_confidence_intervals("ScienceWISE", "uad_vote_None")
    
    #plot_histogram_confidences_interval(["cossim document context vector", "UAD"], [accuracy, accuracy2], [interval, interval2])
    
    #vldb
    #compute_tests_for_technique("ScienceWISE", "cossim_classic_context_vector")
    #compute_tests_for_technique("MSHCorpus", "svm_l2_0.1_0_concat_classic_context_vector_1_doc2vec_25_CBOW_200_2")
    #compute_tests_for_technique("CSWikipedia_res-dup","sci_dr_base_32")
    #compute_tests_for_technique("SDU-AAAI-AD-dedupe","sci_dr_base_32")
    
    #compute_tests_for_technique("ScienceWISE", "cossim_classic_context_vector")
    """
    compute_tests_for_technique("MSHCorpus", "cossim_classic_context_vector", True)
    compute_tests_for_technique("CSWikipedia_res-dup","cossim_classic_context_vector", True)
    compute_tests_for_technique("SDU-AAAI-AD-dedupe","cossim_classic_context_vector", True)
    """

    compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", "_MadDog:TrainIn=Ab3P-BioC_mad_dog_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_1", end_to_end = True)
    #compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", ":TrainIn=Ab3P-BioC_schwartz_hearst_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_1", end_to_end = True)
    #compute_tests_for_technique("Test=UsersWikipedia:TrainOut=FullWikipedia", ":TrainIn=Ab3P-BioC_schwartz_hearst_None_svm_l2_0.01_0_doc2vec_100_CBOW_100_5_0", only_exec = True)

    