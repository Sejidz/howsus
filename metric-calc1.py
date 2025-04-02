import json
import pandas as pd
import numpy as np


def load_files(path):
    dfs = []
    for i in range(30):
        df = pd.read_csv(path+str(i+1)+'.csv')
        dfs.append(df)
    return dfs

# Load the energy and time data from the eb output
# The eb output is a csv file with the energy and time data for each test
def load_eb_output(path):
    df = pd.read_csv(path)
    return df

# Load the memory data from the pyarrow, numpy and pandas test files
# The memory data is stored in the 'USED_MEMORY' column of each test file
def get_avg_mem(dfs):
    avg_mem = []
    for df in dfs:
        avg = df['USED_MEMORY'].mean()
        avg_mem.append(avg)
    return  sum(avg_mem) / len(avg_mem)

# The energy data is stored in the 'joule' column and the time data is stored in the 'time' column
def get_avg_eb(eb):
    avg_energy = eb['Total Energy (J)'].mean()
    avg_time = eb['Spent Time (s)'].mean()
    return avg_energy, avg_time

def calc_component_score(scores, higher_is_better=False):
    minimum = min(scores)
    maximum = max(scores)
    diff = maximum - minimum
    if higher_is_better:
        scoresdist = [((x-minimum)/(diff/3))-1 for x in scores]
    else:
        scoresdist = [((maximum-x)/(diff/3))-1 for x in scores]
    return scoresdist

def get_avgs():
    lightGMFiles = './data/Ml/lightGBM/lightGBM'
    lightGMSummary = './data/Ml/lightGBM/lightGBM.csv'
    lightGMDf = load_files(lightGMFiles)
    lightGMSmry = load_eb_output(lightGMSummary)
    avg_mem_lightGBM = get_avg_mem(lightGMDf)
    avg_energy_lightGBM, avg_time_lightGBM = get_avg_eb(lightGMSmry)

    pytorchFiles = './data/Ml/pytorch/pytorch'
    pytorchSummary = './data/Ml/pytorch/pytorch.csv'
    pytorchDF = load_files(pytorchFiles)
    pytorchSmry = load_eb_output(pytorchSummary)
    avg_mem_pytorch = get_avg_mem(pytorchDF)
    avg_energy_pytorch, avg_time_pytorch = get_avg_eb(pytorchSmry)


    scikitFiles = './data/Ml/scikit/scikit'
    scikitSummary = './data/Ml/scikit/scikit.csv'
    scikitDF = load_files(scikitFiles)
    scikitSmry = load_eb_output(scikitSummary)
    avg_mem_scikit = get_avg_mem(scikitDF)
    avg_energy_scikit, avg_time_scikit = get_avg_eb(scikitSmry)

    tensorFiles = './data/Ml/tensor/tensor'
    tensorSummary = './data/Ml/tensor/tensor.csv'
    tensorDF = load_files(tensorFiles)
    tensorSmry = load_eb_output(tensorSummary)
    avg_mem_tensorflow = get_avg_mem(tensorDF)
    avg_energy_tensorflow, avg_time_tensorflow = get_avg_eb(tensorSmry)

    xgbostFiles = './data/Ml/xgbost/xgbost'
    xgbostSummary = './data/Ml/xgbost/xgbost.csv'
    xgbostDF = load_files(xgbostFiles)
    xgbostSmry = load_eb_output(xgbostSummary)
    avg_mem_xgbost = get_avg_mem(xgbostDF)
    avg_energy_xgbost, avg_time_xgbost = get_avg_eb(xgbostSmry)


    #Accuracu of the models
    #lightGBM:0.9540
    #pytorch: 0.4981
    #scikit: 0.8210
    #tensor: 0.8185
    #xgboost: 0.9170

    avg_mem = [avg_mem_lightGBM, avg_mem_pytorch, avg_mem_scikit, avg_mem_tensorflow, avg_mem_xgbost]
    avg_energy = [avg_energy_lightGBM, avg_energy_pytorch, avg_energy_scikit, avg_energy_tensorflow, avg_energy_xgbost]
    avg_time = [avg_time_lightGBM, avg_time_pytorch, avg_time_scikit, avg_time_tensorflow, avg_time_xgbost]
    avg_acc = [0.9540, 0.4981, 0.8210, 0.8185, 0.9170]
    
    return avg_mem, avg_energy, avg_time, avg_acc

def get_code_metrics(lib_names, json_path):
    with open(json_path) as json_file:
        code_metrics = json.load(json_file)
    
    abstractness = [code_metrics[lib]['abstractness'] for lib in lib_names]
    instability = [code_metrics[lib]['instability'] for lib in lib_names]
    merged_prs = [code_metrics[lib]['merged_prs'] for lib in lib_names]
    closed_issues = [code_metrics[lib]['closed_issues'] for lib in lib_names]
    vulnerabilities = [
        code_metrics[lib]['crit_vulns'] 
        + 0.5*code_metrics[lib]['high_vulns'] 
        + 0.25*code_metrics[lib]['medium_vulns'] 
        + 0.2*code_metrics[lib]['low_vulns'] for lib in lib_names
    ]
    return abstractness, instability, merged_prs, closed_issues, vulnerabilities

def calc_score(lib_names, scores, weights):
    num_libs = len(lib_names)
    num_metrics = len(scores)
    nutris = {}
    for i in range(num_libs):
        lib_scores = [scores[j][i] for j in range(num_metrics)]
        avg = sum([lib_scores[j] * weights[j] for j in range(num_metrics)])/sum(weights)
        nutri = np.select(
            [avg < -0.4,
            (avg >= -0.4) & (avg < 0.2),
            (avg >= 0.2) & (avg < 0.8),
            (avg >= 0.8) & (avg < 1.4),
            avg > 1.4],
            ["E", "D", "C", "B", "A"],
            ""
        )
        nutris[lib_names[i]] = nutri[()]
    print(nutris)
    return nutris

def create_nutris():
    #lib_names = ["lightgbm", "pytorch", "scikit-learn", "tensorflow", "xgboost"]
    lib_names = ["lightgbm", "pytorch", "scikit-learn", "tensorflow"]
    scores_mem, scores_energy, scores_time, scores_acc = get_avgs()
    abst, inst, prs, issues, vulns = get_code_metrics(lib_names, "github-metrics.json")

    mem = calc_component_score(scores_mem)
    energy = calc_component_score(scores_energy)
    time = calc_component_score(scores_time)
    acc = calc_component_score(scores_acc, higher_is_better=True)
    abstractness = calc_component_score(abst, higher_is_better=True)
    instability = calc_component_score(inst)
    merged_prs = calc_component_score(prs, higher_is_better=True)
    closed_issues = calc_component_score(issues, higher_is_better=True)
    vulnerabilities = calc_component_score(vulns)
    
    metrics_to_include = [
        mem,
        energy,
        time,
        acc,
        abstractness,
        instability,
        merged_prs,
        closed_issues,
        vulnerabilities
    ]

    metric_names = ["memory", "energy", "time", "accuracy", "abstractness", "instability", "merged_prs", "closed_issues", "vulnerabilities"]

    weights = {
        "memory": 1,
        "energy": 1,
        "time": 1,
        "accuracy": 2,
        "abstractness": 1,
        "instability": 1,
        "merged_prs": 1,
        "closed_issues": 1,
        "vulnerabilities": 1
    }
    

    return calc_score(lib_names, metrics_to_include, [weights[metric] for metric in metric_names])

print(create_nutris())
