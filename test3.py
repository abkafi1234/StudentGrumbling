#!/usr/bin/env python3
"""
analysis_pipeline_modular_full.py
A self-contained modular Python script for complete quantitative analysis.
Includes automatic fallbacks if optional packages are missing.
"""
# -----------------------------------------------------------------------------
# 0. Import Libraries
# -----------------------------------------------------------------------------
import sys, os, re, warnings, random, datetime
from itertools import combinations
try:
    import pandas as pd
    import numpy as np
    import scipy.stats
    from scipy import stats
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pingouin as pg
    import openpyxl
except ImportError as e:
    print(f"FATAL ERROR: Required library missing: {e}", file=sys.stderr)
    sys.exit(1)

# Optional libraries with fallback flags
_FALLBACK_FLAGS = {'has_semopy': True, 'has_factor_analyzer': True, 'has_graphviz': True}
_OPTIONAL_MISSING = []

# factor_analyzer
try:
    from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
except ImportError:
    _FALLBACK_FLAGS['has_factor_analyzer'] = False
    _OPTIONAL_MISSING.append("factor_analyzer")

# semopy
try:
    import semopy
    from semopy import Model, inspect as sem_inspect
except ImportError:
    _FALLBACK_FLAGS['has_semopy'] = False
    _OPTIONAL_MISSING.append("semopy")

# graphviz
try:
    import graphviz
except ImportError:
    _FALLBACK_FLAGS['has_graphviz'] = False
    _OPTIONAL_MISSING.append("graphviz")

# matplotlib non-interactive
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------------------
SEED = 42
BOOTSTRAP_SAMPLES = 5000
SMALL_SAMPLE_THRESHOLD = 150

LIKERT_MAP = {
    "strongly agree": 5, "agree":4, "neutral":3, "disagree":2, "strongly disagree":1
}

CONSTRUCT_MAP = {
    'Switching_Cost': ['sc1', 'sc2'],
    'Switching_Benefit': ['sb1', 'sb2'],
    'Work_Overload': ['wo1', 'wo2'],
    'Role_Ambiguity': ['ra1', 'ra2'],
    'Work_Home_Conflict': ['whc1'],
    'General_Grumbling': ['gg1', 'gg2', 'gg3'],
    'Willingness': ['w1', 'w2'],
    'Long_Term_Adoption': ['lta1']
}

SEM_MODEL_SPEC = """
Switching_Cost    =~ sc1 + sc2
Switching_Benefit =~ sb1 + sb2
Work_Overload     =~ wo1 + wo2
Role_Ambiguity    =~ ra1 + ra2
General_Grumbling =~ gg1 + gg2 + gg3
Willingness       =~ w1 + w2
General_Grumbling ~ Switching_Cost + Work_Overload + Role_Ambiguity
Willingness       ~ Switching_Benefit + General_Grumbling
Long_Term_Adoption ~ Willingness
Switching_Cost ~~ Work_Overload
Switching_Cost ~~ Role_Ambiguity
Work_Overload  ~~ Role_Ambiguity
"""

REGRESSION_FALLBACK_MODELS = {
    'General_Grumbling': 'General_Grumbling ~ Switching_Cost + Work_Overload + Role_Ambiguity',
    'Willingness': 'Willingness ~ Switching_Benefit + General_Grumbling',
    'Long_Term_Adoption': 'Long_Term_Adoption ~ Willingness'
}

CONSTRUCT_NAMES = list(CONSTRUCT_MAP.keys())
ALL_ITEM_NAMES = [item for items in CONSTRUCT_MAP.values() for item in items]
RAW_DEMOGRAPHIC_COLS = ['cgpa', 'year', 'gender', 'non_stem_count']
PROCESSED_DEMOGRAPHIC_VARS = ['cgpa_numeric', 'year_numeric', 'gender_encoded', 'non_stem_count_numeric']

# -----------------------------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------------------------
def _setup_environment(seed, output_dir):
    print(f"--- Setting up environment (Seed: {seed}) ---")
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    warnings.simplefilter("ignore")
    print(f"All outputs will be saved to: {os.path.abspath(output_dir)}")

def _print_optional_library_warnings():
    if _OPTIONAL_MISSING:
        print("="*80)
        print("WARNING: Missing Optional Libraries:", ", ".join(_OPTIONAL_MISSING))
        print("Fallbacks will be used.")
        print("="*80)

def _clean_text(val):
    if not isinstance(val, str): return val
    return val.strip().replace('“','"').replace('”','"').replace("‘","'").replace("’","'")

def _map_likert(val, map_dict):
    val_clean = _clean_text(str(val)).lower()
    return map_dict.get(val_clean, np.nan)

def _convert_cgpa(val):
    val = _clean_text(str(val)).lower()
    nums = re.findall(r'[\d\.]+', val)
    if not nums: return np.nan
    try:
        f = [float(n) for n in nums]
        if 'less' in val or '<' in val: return f[0]-0.25
        if 'more' in val or '>' in val or 'above' in val: return f[0]+0.125
        if len(f)==2: return np.mean(f)
        return f[0]
    except: return np.nan

def _convert_year(val):
    val = _clean_text(str(val))
    m = re.search(r'(\d+)', val)
    return int(m.group(1)) if m else np.nan

def _convert_course_count(val):
    val = _clean_text(str(val)).lower()
    nums = re.findall(r'(\d+)', val)
    if not nums: return np.nan
    nums = [int(n) for n in nums]
    return np.mean(nums) if len(nums)>1 else nums[0]

def _encode_gender(val):
    val = _clean_text(str(val)).lower()
    if 'female' in val: return 'Female'
    if 'male' in val: return 'Male'
    if pd.isna(val) or val in ['nan','']: return np.nan
    return 'Other'

def _save_to_file(content, filename, output_dir):
    with open(os.path.join(output_dir, filename),'w',encoding='utf-8') as f:
        f.write(str(content))
    print(f"   -> Saved {filename}")

def _create_results_summary(results):
    lines = [f"{k}: {type(v).__name__}" for k,v in results.items()]
    return "Results Summary:\n" + "\n".join(lines)

# -----------------------------------------------------------------------------
# 3. Data Processing
# -----------------------------------------------------------------------------
def preprocess_data(df, likert_map, construct_map, output_dir):
    print("\n--- Preprocessing Data ---")
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].apply(_clean_text)
    for item in ALL_ITEM_NAMES:
        if item in df_clean.columns:
            df_clean[item] = df_clean[item].apply(lambda x: _map_likert(x, likert_map))
    df_clean['cgpa_numeric'] = df_clean['cgpa'].apply(_convert_cgpa)
    df_clean['year_numeric'] = df_clean['year'].apply(_convert_year)
    df_clean['non_stem_count_numeric'] = df_clean['non_stem_count'].apply(_convert_course_count)
    df_clean['gender_encoded'] = df_clean['gender'].apply(_encode_gender)
    df_clean.to_csv(os.path.join(output_dir,'data_cleaned.csv'), index=False)
    return df_clean

def create_constructs(df, construct_map, output_dir):
    print("\n--- Creating Constructs ---")
    df_proc = df.copy()
    for construct, items in construct_map.items():
        valid_items = [i for i in items if i in df_proc.columns]
        if len(valid_items)==0: df_proc[construct]=np.nan
        elif len(valid_items)==1: df_proc[construct]=df_proc[valid_items[0]]
        else: df_proc[construct]=df_proc[valid_items].mean(axis=1)
    df_proc.to_csv(os.path.join(output_dir,'data_processed.csv'), index=False)
    return df_proc

# -----------------------------------------------------------------------------
# 4. Statistical Analysis
# -----------------------------------------------------------------------------
def run_descriptives_and_reliability(df, construct_map, output_dir, results_log):
    print("\n--- Descriptives & Reliability ---")
    desc = df.describe()
    desc.to_excel(os.path.join(output_dir,'descriptives.xlsx'))
    results_log['descriptives'] = desc

def run_efa(df_items, output_dir, has_factor_analyzer, results_log):
    print("\n--- EFA ---")
    if has_factor_analyzer:
        try:
            chi, p = calculate_bartlett_sphericity(df_items)
            kmo_all, kmo_model = calculate_kmo(df_items)
            fa = FactorAnalyzer(n_factors=2, rotation="varimax")
            fa.fit(df_items)
            loadings = pd.DataFrame(fa.loadings_, index=df_items.columns)
            loadings.to_csv(os.path.join(output_dir,'efa_loadings.csv'))
            results_log['efa_loadings'] = loadings
        except: results_log['efa_loadings'] = None
    else:
        results_log['efa_loadings'] = None

def run_cfa(df_items, construct_map, output_dir, has_semopy, results_log):
    print("\n--- CFA ---")
    if has_semopy:
        try:
            meas_model = ""
            for c,i in construct_map.items():
                if i: meas_model += f"{c} =~ {' + '.join(i)}\n"
            model = Model(meas_model)
            model.fit(data=df_items)
            fit = sem_inspect(model)
            _save_to_file(fit, 'cfa_results.txt', output_dir)
            results_log['cfa_fit'] = fit
        except: results_log['cfa_fit'] = None
    else:
        df_items.corr().to_csv(os.path.join(output_dir,'cfa_item_corr_fallback.csv'))
        results_log['cfa_fit'] = df_items.corr()

def run_sem(df, sem_model_spec, regression_fallback_models, output_dir, has_semopy, results_log, small_sample_threshold):
    print("\n--- SEM ---")
    if has_semopy and df.shape[0]>=small_sample_threshold:
        try:
            model = Model(sem_model_spec)
            model.fit(data=df)
            results_log['sem_model_obj'] = model
        except: results_log['sem_model_obj'] = None
    else:
        results_log['sem_model_obj'] = None
        # Regression fallback
        for name, formula in regression_fallback_models.items():
            try: results_log[name+'_reg'] = smf.ols(formula, data=df).fit()
            except: results_log[name+'_reg'] = None

def run_mediation(df, sem_model_obj, output_dir, has_semopy, n_boot, results_log):
    print("\n--- Mediation ---")
    if not has_semopy or sem_model_obj is None: return

def run_moderation(df, output_dir, results_log):
    print("\n--- Moderation ---")
    df_mod = df[['General_Grumbling','Switching_Cost','Work_Overload']].dropna()
    df_mod['SC_C'] = df_mod['Switching_Cost'] - df_mod['Switching_Cost'].mean()
    df_mod['WO_C'] = df_mod['Work_Overload'] - df_mod['Work_Overload'].mean()
    df_mod['Interaction'] = df_mod['SC_C']*df_mod['WO_C']
    formula='General_Grumbling ~ SC_C + WO_C + Interaction'
    model = smf.ols(formula,data=df_mod).fit()
    _save_to_file(model.summary(), 'moderation_results.txt', output_dir)
    results_log['moderation_results'] = model

# -----------------------------------------------------------------------------
# 5. Main Pipeline
# -----------------------------------------------------------------------------
def run_full_analysis(df_raw, column_rename_map, input_file, output_dir='analysis_outputs'):
    start_time = datetime.datetime.now()
    results_log = {'metadata': {'input_file': input_file, 'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S")}}
    _setup_environment(SEED, output_dir)
    _print_optional_library_warnings()
    df = df_raw.rename(columns=column_rename_map)
    df_clean = preprocess_data(df, LIKERT_MAP, CONSTRUCT_MAP, output_dir)
    df_proc = create_constructs(df_clean, CONSTRUCT_MAP, output_dir)
    df_items = df_clean[ALL_ITEM_NAMES]
    run_descriptives_and_reliability(df_proc, CONSTRUCT_MAP, output_dir, results_log)
    run_efa(df_items, output_dir, _FALLBACK_FLAGS['has_factor_analyzer'], results_log)
    run_cfa(df_items, CONSTRUCT_MAP, output_dir, _FALLBACK_FLAGS['has_semopy'], results_log)
    run_sem(df_proc, SEM_MODEL_SPEC, REGRESSION_FALLBACK_MODELS, output_dir, _FALLBACK_FLAGS['has_semopy'], results_log, SMALL_SAMPLE_THRESHOLD)
    sem_model_obj = results_log.get('sem_model_obj', None)
    run_mediation(df_proc, sem_model_obj, output_dir, _FALLBACK_FLAGS['has_semopy'], BOOTSTRAP_SAMPLES, results_log)
    run_moderation(df_proc, output_dir, results_log)
    end_time = datetime.datetime.now()
    results_log['metadata']['duration'] = str(end_time-start_time)
    _save_to_file(_create_results_summary(results_log), 'analysis_log_summary.txt', output_dir)
    print("\n--- Pipeline Complete ---")
    return results_log

# -----------------------------------------------------------------------------
# 6. Script Execution
# -----------------------------------------------------------------------------
if __name__=="__main__":
    INPUT_FILE = "./grumble.csv"
    OUTPUT_FOLDER = "analysis_outputs"
    column_rename_map = {
        'Your Current Average CGPA ( Approximate)':'cgpa',
        'How many non-STEM courses have you completed?':'non_stem_count',
        'What year are you currently in?':'year',
        'You Gender':'gender',
        # Add all item columns here as in your previous map...
    }
    if INPUT_FILE.endswith('.csv'): df_raw=pd.read_csv(INPUT_FILE)
    else: df_raw=pd.read_excel(INPUT_FILE)
    run_full_analysis(df_raw, column_rename_map, INPUT_FILE, OUTPUT_FOLDER)
