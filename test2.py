#!/usr/bin/env python3
"""
analysis_pipeline_modular.py
A modular Python script for a complete quantitative analysis pipeline.
This script is refactored to be used as a module. You can load your
data using pandas, ensure the column names match the expected input,
and then call the `run_full_analysis()` function to execute the pipeline.
"""
# -----------------------------------------------------------------------------
# 0. Import Libraries
# -----------------------------------------------------------------------------
import sys
import os
import re
import warnings
import random
import datetime
from itertools import combinations
# --- Required Libraries ---
try:
    import pandas as pd
    import numpy as np
    import scipy.stats
    from scipy import stats
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import statsmodels
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.iolib.summary2 import summary_col
    import pingouin as pg
    import openpyxl
except ImportError as e:
    print(f"FATAL ERROR: A required library is missing: {e}", file=sys.stderr)
    print("Please install all required libraries:", file=sys.stderr)
    print("pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels pingouin openpyxl", file=sys.stderr)
    sys.exit(1)
# --- Optional Libraries (with fallbacks) ---
_FALLBACK_FLAGS = {
    'has_semopy': True,
    'has_factor_analyzer': True,
    'has_graphviz': True
}
_OPTIONAL_MISSING = []
try:
    import factor_analyzer
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import (
        calculate_bartlett_sphericity, calculate_kmo
    )
except ImportError:
    _FALLBACK_FLAGS['has_factor_analyzer'] = False
    _OPTIONAL_MISSING.append("factor_analyzer")
try:
    import semopy
    from semopy import Model
    from semopy import inspect as sem_inspect
    from semopy import report as sem_report
    from semopy import compare as sem_compare
except ImportError:
    _FALLBACK_FLAGS['has_semopy'] = False
    _OPTIONAL_MISSING.append("semopy")
try:
    import graphviz
    # Test if Graphviz executable is in path
    try:
        os.makedirs(os.path.join(os.getcwd(), 'outputs'), exist_ok=True)
        graphviz.Source('digraph G {}').render(directory=os.path.join(os.getcwd(), 'outputs'), filename='gv_test', cleanup=True)
        if os.path.exists(os.path.join(os.getcwd(), 'outputs', 'gv_test')):
            os.remove(os.path.join(os.getcwd(), 'outputs', 'gv_test'))
    except (graphviz.backend.execute.CalledProcessError, FileNotFoundError):
        _FALLBACK_FLAGS['has_graphviz'] = False
        _OPTIONAL_MISSING.append("graphviz (and the system executable)")
except ImportError:
    _FALLBACK_FLAGS['has_graphviz'] = False
    _OPTIONAL_MISSING.append("graphviz")
# Use a non-interactive backend for script-based plotting
matplotlib.use('Agg')
# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
# --- EDIT THESE VARIABLES TO MATCH YOUR DATASET AND MODELS ---
# --- Analysis Settings ---
SEED = 42
BOOTSTRAP_SAMPLES = 5000  # Number of bootstrap samples for mediation
SMALL_SAMPLE_THRESHOLD = 150  # n < this triggers fallback from SEM
# --- Data Mapping ---
# Likert scale text-to-numeric mapping
LIKERT_MAP = {
    "Strongly agree": 5,
    "Agree": 4,
    "Neutral": 3,
    "Disagree": 2,
    "Strongly disagree": 1,
    # Add common variations
    "strongly agree": 5,
    "agree": 4,
    "neutral": 3,
    "disagree": 2,
    "strongly disagree": 1,
}
# Defines which items form which construct.
# !! YOUR INPUT DATAFRAME MUST HAVE THESE COLUMN NAMES !!
CONSTRUCT_MAP = {
    'Switching_Cost': ['sc1', 'sc2'],
    'Switching_Benefit': ['sb1', 'sb2'],
    'Work_Overload': ['wo1', 'wo2'],
    'Role_Ambiguity': ['ra1', 'ra2'],
    'Work_Home_Conflict': ['whc1'],  # Single item
    'General_Grumbling': ['gg1', 'gg2', 'gg3'],
    'Willingness': ['w1', 'w2'],
    'Long_Term_Adoption': ['lta1']  # Single item
}
# --- Model Specification ---
# SEM model structure (used by semopy)
SEM_MODEL_SPEC = """
# Measurement Model (CFA part)
Switching_Cost    =~ sc1 + sc2
Switching_Benefit =~ sb1 + sb2
Work_Overload     =~ wo1 + wo2
Role_Ambiguity    =~ ra1 + ra2
General_Grumbling =~ gg1 + gg2 + gg3
Willingness       =~ w1 + w2
# Structural Model (Path analysis part)
General_Grumbling ~ Switching_Cost + Work_Overload + Role_Ambiguity
Willingness       ~ Switching_Benefit + General_Grumbling
Long_Term_Adoption ~ Willingness
# Latent variable covariances (optional but good practice)
Switching_Cost ~~ Work_Overload
Switching_Cost ~~ Role_Ambiguity
Work_Overload  ~~ Role_Ambiguity
"""
# Regression-based fallback models
REGRESSION_FALLBACK_MODELS = {
    'General_Grumbling': 'General_Grumbling ~ Switching_Cost + Work_Overload + Role_Ambiguity',
    'Willingness': 'Willingness ~ Switching_Benefit + General_Grumbling',
    'Long_Term_Adoption': 'Long_Term_Adoption ~ Willingness'
}
# --- Generated Column Lists (Do not edit) ---
# List of all construct names
CONSTRUCT_NAMES = list(CONSTRUCT_MAP.keys())
# List of all item names
ALL_ITEM_NAMES = [item for items in CONSTRUCT_MAP.values() for item in items]
# List of demographic/control variables used in preprocessing
# !! YOUR INPUT DATAFRAME MUST ALSO HAVE THESE COLUMN NAMES !!
RAW_DEMOGRAPHIC_COLS = ['cgpa', 'year', 'gender', 'non_stem_count']
# List of processed demographic/control variables for analysis
PROCESSED_DEMOGRAPHIC_VARS = ['cgpa_numeric', 'year_numeric', 'gender_encoded', 'non_stem_count_numeric']
# ---
# !! This is the full list of columns your input DataFrame must provide !!
REQUIRED_INPUT_COLUMNS = sorted(list(set(ALL_ITEM_NAMES + RAW_DEMOGRAPHIC_COLS)))
# ---
# -----------------------------------------------------------------------------
# 2. Helper Functions (Internal)
# -----------------------------------------------------------------------------
def _setup_environment(seed, output_dir):
    """
    Creates output directory, sets random seeds, and configures warnings.
    """
    print(f"--- Setting up environment (Seed: {seed}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Suppress common warnings for cleaner output
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
    except ImportError:
        pass
    
    print(f"All outputs will be saved to: {os.path.abspath(output_dir)}")
def _print_optional_library_warnings():
    """Prints warnings if optional libraries are missing."""
    if _OPTIONAL_MISSING:
        print("="*80)
        print("WARNING: Missing Optional Libraries")
        print("The following optional libraries are not installed:")
        print(f"  {', '.join(_OPTIONAL_MISSING)}")
        print("The script will run, but will use fallback analyses:")
        if "factor_analyzer" in _OPTIONAL_MISSING:
            print("  - EFA will use scikit-learn PCA instead of FactorAnalyzer.")
        if "semopy" in _OPTIONAL_MISSING:
            print("  - CFA/SEM/Multi-group analysis will use regression-based fallbacks.")
        if "graphviz" in _OPTIONAL_MISSING:
            print("  - SEM path diagram will not be generated.")
        print(f"\nTo install them, run (example):")
        print(f"  pip install {' '.join(_OPTIONAL_MISSING)}")
        print("="*80)
def _clean_text(cell_value):
    """Helper to normalize text values."""
    if not isinstance(cell_value, str):
        return cell_value
    return cell_value.strip().replace('“', '"').replace('”', '"') \
                     .replace("‘", "'").replace("’", "'")
def _map_likert(cell_value, likert_map):
    """Helper to map Likert text to numbers."""
    cleaned = _clean_text(cell_value)
    if cleaned is None:
        return np.nan
    return likert_map.get(cleaned.lower(), np.nan)
def _convert_cgpa(cell_value):
    """Converts text CGPA ranges to a numeric midpoint."""
    cleaned = _clean_text(str(cell_value)).lower()
    numbers = re.findall(r'[\d\.]+', cleaned)
    if not numbers:
        return np.nan
    try:
        floats = [float(n) for n in numbers]
        if 'less than' in cleaned or '<' in cleaned:
            return floats[0] - 0.25
        elif 'more than' in cleaned or '>' in cleaned or 'above' in cleaned:
            return floats[0] + 0.125
        elif len(floats) == 2:
            return np.mean(floats)
        elif len(floats) == 1:
            return floats[0]
        else:
            return np.nan
    except (ValueError, IndexError):
        return np.nan
def _convert_year(cell_value):
    """Converts text year (e.g., "4th year") to integer 4."""
    cleaned = _clean_text(str(cell_value))
    match = re.search(r'(\d)', cleaned)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return np.nan
    return np.nan
def _convert_course_count(cell_value):
    """Converts text count (e.g., "About 10", "5-6") to numeric."""
    cleaned = _clean_text(str(cell_value)).lower()
    numbers = re.findall(r'(\d+)', cleaned)
    if not numbers:
        return np.nan
    try:
        ints = [int(n) for n in numbers]
        if len(ints) == 2:
            return np.mean(ints)
        elif len(ints) == 1:
            return ints[0]
        else:
            return np.nan
    except (ValueError, IndexError):
        return np.nan
def _encode_gender(cell_value):
    """Encodes gender into 'Male', 'Female', 'Other'."""
    cleaned = _clean_text(str(cell_value)).lower()
    if 'female' in cleaned:
        return 'Female'
    elif 'male' in cleaned:
        return 'Male'
    elif pd.isna(cleaned) or cleaned in ['nan', '']:
        return np.nan
    else:
        return 'Other'
def _save_to_file(content, filename, output_dir):
    """Helper to save text content to a file."""
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   -> Saved {filename}")
    except Exception as e:
        print(f"   ! ERROR: Failed to save {filename}. Reason: {e}")
def _create_results_summary(results_dict):
    """Helper to create a simple text summary from a results dictionary."""
    summary_lines = [
        f"Key: '{key}' | Type: {type(value).__name__}" 
        for key, value in results_dict.items()
    ]
    return "Summary of collected results objects:\n" + "\n".join(summary_lines)
# -----------------------------------------------------------------------------
# 3. Analysis Pipeline Functions
# -----------------------------------------------------------------------------
def preprocess_data(df, likert_map, construct_map, output_dir):
    """
    Applies all cleaning and transformation rules to the data.
    Saves the cleaned (but pre-construct) data.
    Assumes `df` contains all required columns.
    """
    print(f"\n--- 1. Preprocessing Data ---")
    
    # Check for required columns
    all_req_cols = set(ALL_ITEM_NAMES + RAW_DEMOGRAPHIC_COLS)
    missing_cols = all_req_cols - set(df.columns)
    if missing_cols:
        print("="*80, file=sys.stderr)
        print("FATAL ERROR: Missing Required Columns in DataFrame", file=sys.stderr)
        print("The provided DataFrame is missing the following columns:", file=sys.stderr)
        print(f"  {', '.join(sorted(list(missing_cols)))}", file=sys.stderr)
        print("\nPlease ensure your DataFrame is loaded and columns are renamed", file=sys.stderr)
        print("to match the names in REQUIRED_INPUT_COLUMNS.", file=sys.stderr)
        print("="*80, file=sys.stderr)
        sys.exit(1)
    df_clean = df.copy()
    # 1. Clean all text cells (object columns) first
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].apply(_clean_text)
    # 2. Map Likert items
    item_cols = [item for items in construct_map.values() for item in items]
    for col in item_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(lambda x: _map_likert(x, likert_map))
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    print(f"Mapped {len(item_cols)} Likert items to numeric scale.")
    # 3. Convert demographic/control columns
    # These functions expect columns 'cgpa', 'year', 'non_stem_count', 'gender'
    df_clean['cgpa_numeric'] = df_clean['cgpa'].apply(_convert_cgpa)
    df_clean['year_numeric'] = df_clean['year'].apply(_convert_year)
    df_clean['non_stem_count_numeric'] = df_clean['non_stem_count'].apply(_convert_course_count)
    df_clean['gender_encoded'] = df_clean['gender'].apply(_encode_gender)
    print("Converted CGPA, Year, Course Count, and Gender columns.")
    # 4. Save cleaned data
    out_path = os.path.join(output_dir, 'data_cleaned.csv')
    df_clean.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"   -> Saved data_cleaned.csv")
    
    return df_clean
def create_constructs(df, construct_map, output_dir):
    """
    Creates composite construct scores from items.
    Saves the final processed data with constructs.
    """
    print(f"\n--- 2. Creating Construct Scores ---")
    df_processed = df.copy()
    
    for construct, items in construct_map.items():
        if not items:
            continue
        
        valid_items = [item for item in items if item in df_processed.columns]
        if not valid_items:
            print(f"   ! WARNING: No valid items found for construct '{construct}'. Skipping.")
            df_processed[construct] = np.nan
            continue
        
        if len(valid_items) < len(items):
             print(f"   ! WARNING: Missing items for construct '{construct}'. "
                   f"Using: {', '.join(valid_items)}")
        if len(valid_items) == 1:
            df_processed[construct] = df_processed[valid_items[0]]
        else:
            item_data = df_processed[valid_items]
            df_processed[construct] = item_data.mean(axis=1, skipna=True)
            df_processed[construct] = df_processed[construct].where(
                item_data.notna().any(axis=1), 
                np.nan
            )
    
    print(f"Created {len(construct_map)} construct composite scores.")
    
    # Save processed data
    out_path = os.path.join(output_dir, 'data_processed.csv')
    df_processed.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"   -> Saved data_processed.csv")
    
    return df_processed
def run_descriptives_and_reliability(df_processed, construct_map, output_dir, results_log):
    """
    3A. Calculates and saves descriptive statistics and Cronbach's alpha.
    """
    print(f"\n--- 3A. Descriptives & Reliability ---")
    
    # --- Descriptives ---
    item_cols = [item for items in construct_map.values() for item in items]
    construct_cols = list(construct_map.keys())
    
    cols_to_describe = item_cols + construct_cols + PROCESSED_DEMOGRAPHIC_VARS
    cols_to_describe = sorted(list(set(
        [col for col in cols_to_describe if col in df_processed.columns]
    )))
    
    desc_stats = df_processed[cols_to_describe].describe().T
    desc_stats = desc_stats[['count', 'mean', 'std', 'min', 'max']]
    desc_stats['N'] = desc_stats['count'].astype(int)
    desc_stats = desc_stats[['N', 'mean', 'std', 'min', 'max']]
    
    out_path = os.path.join(output_dir, 'descriptives.xlsx')
    desc_stats.to_excel(out_path)
    print(f"   -> Saved descriptives.xlsx")
    results_log['descriptives'] = desc_stats
    
    # --- Reliability (Cronbach's Alpha) ---
    reliability_results = []
    alpha_warnings = []
    
    for construct, items in construct_map.items():
        if len(items) < 2:
            continue  # Alpha not defined for single-item constructs
            
        item_data = df_processed[items].dropna(how='all')
        
        if item_data.shape[0] < 2 or item_data.shape[1] < 2:
            continue
            
        try:
            alpha_stats = pg.reliability(data=item_data.dropna(how='any'))
            alpha_val = alpha_stats.loc['alpha', 'alpha']
            alpha_ci = alpha_stats.loc['alpha', 'CI95%']
            if_deleted = alpha_stats.loc['alpha-if-deleted'].to_dict()
            
            reliability_results.append({
                'Construct': construct,
                'N_Items': len(items),
                'N_Obs': item_data.dropna(how='any').shape[0],
                'Alpha': alpha_val,
                '95%_CI_Lower': alpha_ci[0],
                '95%_CI_Upper': alpha_ci[1]
            })
            
            if alpha_val < 0.70:
                warning_msg = (
                    f"   ! RELIABILITY WARNING for '{construct}': "
                    f"Alpha = {alpha_val:.3f} (below 0.70)."
                )
                print(warning_msg)
                alpha_warnings.append(warning_msg)
                for item, if_del_alpha in if_deleted.items():
                    if if_del_alpha > alpha_val:
                        suggestion = f"     - Consider dropping '{item}' (Alpha would be {if_del_alpha:.3f})"
                        print(suggestion)
                        alpha_warnings.append(suggestion)
        except Exception as e:
            print(f"   ! ERROR calculating reliability for '{construct}': {e}")
            
    if reliability_results:
        reliability_df = pd.DataFrame(reliability_results).set_index('Construct')
        out_path = os.path.join(output_dir, 'reliability.csv')
        reliability_df.to_csv(out_path)
        print(f"   -> Saved reliability.csv")
        results_log['reliability'] = reliability_df
    else:
        print("   No reliability analyses were run (e.g., all single-item constructs).")
    results_log['reliability_warnings'] = "\n".join(alpha_warnings)
def run_efa(df_items, output_dir, has_factor_analyzer, results_log):
    """
    3B. Runs Exploratory Factor Analysis (EFA) or PCA fallback.
    """
    print(f"\n--- 3B. Exploratory Factor Analysis (EFA) ---")
    
    df_efa = df_items.dropna()
    n_obs, n_vars = df_efa.shape
    
    if n_obs < n_vars:
        print(f"   ! WARNING: More variables ({n_vars}) than observations ({n_obs}). Skipping EFA/PCA.")
        results_log['efa_loadings'] = None
        results_log['efa_message'] = "Skipped: More variables than observations."
        return
    if n_obs < 50:
        print(f"   ! WARNING: Very small sample size ({n_obs}) for EFA/PCA.")
    if has_factor_analyzer:
        try:
            print("   Running EFA with 'factor_analyzer' (Varimax rotation).")
            chi_square_value, p_value = calculate_bartlett_sphericity(df_efa)
            kmo_all, kmo_model = calculate_kmo(df_efa)
            print(f"   Bartlett's Test: chi2={chi_square_value:.3f}, p={p_value:.3e}")
            print(f"   KMO Test: {kmo_model:.3f}")
            
            fa_check = FactorAnalyzer(n_factors=n_vars, rotation=None)
            fa_check.fit(df_efa)
            ev, _ = fa_check.get_eigenvalues()
            n_factors = (ev > 1).sum()
            if n_factors == 0:
                n_factors = 1
            print(f"   Determined {n_factors} factors based on Eigenvalue > 1 rule.")
            
            fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
            fa.fit(df_efa)
            
            loadings = fa.loadings_
            communalities = fa.get_communalities()
            
            loadings_df = pd.DataFrame(
                loadings, 
                index=df_efa.columns,
                columns=[f'Factor{i+1}' for i in range(n_factors)]
            )
            loadings_df['Communalities'] = communalities
            
            out_path = os.path.join(output_dir, 'efa_loadings.csv')
            loadings_df.to_csv(out_path)
            print(f"   -> Saved efa_loadings.csv")
            results_log['efa_loadings'] = loadings_df
            results_log['efa_message'] = f"Success: 'factor_analyzer' with {n_factors} factors."
        except Exception as e:
            print(f"   ! ERROR during EFA with 'factor_analyzer': {e}. Skipping.")
            results_log['efa_loadings'] = None
            results_log['efa_message'] = f"Error during factor_analyzer: {e}"
    else:
        # --- Fallback to PCA ---
        print("   ! FALLBACK: 'factor_analyzer' not found. Running PCA.")
        try:
            scaler = StandardScaler()
            df_efa_scaled = scaler.fit_transform(df_efa)
            
            pca_check = PCA(n_components=None)
            pca_check.fit(df_efa_scaled)
            n_components = (pca_check.explained_variance_ > 1).sum()
            if n_components == 0:
                n_components = 1
            print(f"   Determined {n_components} components based on Eigenvalue > 1 rule.")
            
            pca = PCA(n_components=n_components)
            pca.fit(df_efa_scaled)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            
            loadings_df = pd.DataFrame(
                loadings, 
                index=df_efa.columns,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            out_path = os.path.join(output_dir, 'efa_loadings_pca_fallback.csv')
            loadings_df.to_csv(out_path)
            print(f"   -> Saved efa_loadings_pca_fallback.csv")
            results_log['efa_loadings'] = loadings_df
            results_log['efa_message'] = f"Fallback: PCA with {n_components} components."
        except Exception as e:
            print(f"   ! ERROR during PCA fallback: {e}. Skipping.")
            results_log['efa_loadings'] = None
            results_log['efa_message'] = f"Error during PCA fallback: {e}"
def run_cfa(df_items, construct_map, output_dir, has_semopy, results_log):
    """
    3C. Runs Confirmatory Factor Analysis (CFA) or fallback.
    """
    print(f"\n--- 3C. Confirmatory Factor Analysis (CFA) ---")
    
    df_cfa = df_items.dropna()
    n_obs = df_cfa.shape[0]
    if n_obs < 100:
        print(f"   ! WARNING: Small sample size ({n_obs}) for CFA. Fit indices may be unreliable.")
    
    meas_model_spec = ""
    for construct, items in construct_map.items():
        if len(items) > 0:
             meas_model_spec += f"{construct} =~ {' + '.join(items)}\n"
    
    constructs_with_items = [c for c, i in construct_map.items() if len(i) > 0]
    for c1, c2 in combinations(constructs_with_items, 2):
        meas_model_spec += f"{c1} ~~ {c2}\n"
    results_log['cfa_model_spec'] = meas_model_spec
    
    if has_semopy:
        try:
            print("   Running CFA with 'semopy'.")
            model = Model(meas_model_spec)
            model.fit(data=df_cfa)
            
            fit_stats = sem_inspect(model)
            fit_stats_str = str(fit_stats)
            print("   CFA Fit Statistics:")
            print(fit_stats_str)
            
            _save_to_file(fit_stats_str, 'cfa_results.txt', output_dir)
            results_log['cfa_fit'] = fit_stats
            results_log['cfa_message'] = "Success: 'semopy' CFA complete."
        except Exception as e:
            print(f"   ! ERROR during 'semopy' CFA: {e}.")
            _save_to_file(f"semopy CFA failed: {e}", 'cfa_results.txt', output_dir)
            results_log['cfa_fit'] = None
            results_log['cfa_message'] = f"Error: {e}"
            has_semopy = False
    
    if not has_semopy:
        print("   ! FALLBACK: 'semopy' not found or failed. Running correlation-based validity check.")
        item_corr = df_items.corr()
        out_path = os.path.join(output_dir, 'cfa_item_corr_fallback.csv')
        item_corr.to_csv(out_path)
        print(f"   -> Saved cfa_item_corr_fallback.csv")
        fallback_msg = "FALLBACK: 'semopy' not found or failed.\nAn item-item correlation matrix was saved instead."
        _save_to_file(fallback_msg, 'cfa_results.txt', output_dir)
        results_log['cfa_fit'] = item_corr
        results_log['cfa_message'] = "Fallback: Saved item correlation matrix."
def run_sem(df_processed, sem_model_spec, regression_fallback_models,
            output_dir, has_semopy, has_graphviz, results_log, small_sample_threshold):
    """
    3D. Runs Structural Equation Modeling (SEM) or regression fallback.
    """
    print(f"\n--- 3D. Structural Equation Modeling (SEM) ---")
    
    df_sem = df_processed.dropna(subset=ALL_ITEM_NAMES)
    n_obs = df_sem.shape[0]
    
    if n_obs < small_sample_threshold:
        print(f"   ! WARNING: Sample size {n_obs} < {small_sample_threshold}. Forcing regression fallback.")
        has_semopy = False
    
    results_log['sem_model_spec'] = sem_model_spec
    
    if has_semopy:
        try:
            print("   Running full SEM with 'semopy'.")
            model = Model(sem_model_spec)
            model.fit(data=df_sem)
            
            fit_stats = sem_inspect(model)
            fit_stats_str = str(fit_stats)
            print("   SEM Fit Statistics:")
            print(fit_stats_str)
            
            params = sem_inspect(model, 'std_est')
            params_str = str(params)
            
            report_str = f"--- SEM Fit Statistics ---\n{fit_stats_str}\n\n--- Standardized Estimates ---\n{params_str}"
            _save_to_file(report_str, 'sem_results.txt', output_dir)
            
            results_log['sem_fit'] = fit_stats
            results_log['sem_params'] = params
            results_log['sem_model_obj'] = model
            results_log['sem_message'] = "Success: 'semopy' SEM complete."
            
            # Generate graph
            if has_graphviz:
                try:
                    from semopy.visual import semplota
                    semplot(model, os.path.join(output_dir, 'sem_model_diagram.png'))
                    print(f"   -> Saved sem_model_diagram.png")
                except Exception as e:
                    print(f"   ! ERROR: Failed to generate SEM diagram: {e}")
            else:
                print("   Skipping SEM diagram: 'graphviz' not found.")
                
        except Exception as e:
            print(f"   ! ERROR during 'semopy' SEM: {e}. Switching to fallback.")
            results_log['sem_message'] = f"Error: {e}. Switching to fallback."
            has_semopy = False
    
    if not has_semopy:
        print("   ! FALLBACK: 'semopy' not found or failed. Running OLS regression.")
        regression_results = {}
        report_lines = ["--- OLS REGRESSION FALLBACK RESULTS ---"]
        
        # Use construct scores
        df_constructs = df_processed[CONSTRUCT_NAMES].dropna()
        
        for name, formula in regression_fallback_models.items():
            try:
                model = smf.ols(formula, data=df_constructs).fit()
                regression_results[name] = model
                report_lines.append(f"\n\n--- Model: {name} (Dependent: {formula.split('~')[0].strip()}) ---")
                report_lines.append(str(model.summary()))
            except Exception as e:
                print(f"   ! ERROR running regression for '{name}': {e}")
                report_lines.append(f"\n\n--- Model: {name} FAILED ---")
                report_lines.append(str(e))
        
        report_str = "\n".join(report_lines)
        _save_to_file(report_str, 'sem_results_regression_fallback.txt', output_dir)
        results_log['sem_fit'] = None
        results_log['sem_params'] = regression_results
        results_log['sem_model_obj'] = None
        if 'sem_message' not in results_log:
             results_log['sem_message'] = "Fallback: OLS regression complete."
def run_mediation(df_processed, sem_model_obj, output_dir, has_semopy, n_boot, results_log):
    """
    3E. Runs Mediation analysis.
    """
    print(f"\n--- 3E. Mediation Analysis ---")
    
    if not has_semopy or sem_model_obj is None:
        print("   ! SKIPPING: Mediation requires a successful 'semopy' SEM run.")
        results_log['mediation_results'] = None
        results_log['mediation_message'] = "Skipped: 'semopy' was not available."
        return
    
    try:
        print(f"   Running bootstrap mediation with {n_boot} samples...")
        
        # Define the indirect effect(s) based on the model
        # Example: X -> M -> Y is (X->M) * (M->Y)
        # Our model: 
        # 1. Switching_Benefit -> General_Grumbling -> Willingness (Not in model)
        # 2. General_Grumbling -> Willingness -> Long_Term_Adoption
        # Let's test #2
        # `a` = General_Grumbling -> Willingness
        # `b` = Willingness -> Long_Term_Adoption
        # We need the parameter names.
        
        # Let's inspect param names from the model
        params = sem_inspect(sem_model_obj)
        param_names = params.index.to_list()
        
        # Find the param names. This is fragile but necessary.
        # This assumes param names like 'Willingness~General_Grumbling'
        try:
            p_a = [p for p in param_names if p.startswith('Willingness') and p.endswith('General_Grumbling')][0]
            p_b = [p for p in param_names if p.startswith('Long_Term_Adoption') and p.endswith('Willingness')][0]
        except IndexError:
            print("   ! ERROR: Could not find mediation path parameters in SEM model.")
            print("     Expected 'Willingness~General_Grumbling' and 'Long_Term_Adoption~Willingness'.")
            results_log['mediation_results'] = None
            results_log['mediation_message'] = "Error: Could not find mediation path parameters."
            return
            
        effects_def = {
            'indirect_effect': (p_a, p_b)
        }
        
        boot_results = sem_model_obj.bootstrap(n_boot=n_boot, effects=effects_def)
        
        # Get the summary for the indirect effect
        indirect_summary = boot_results['indirect_effect']
        
        report_str = f"--- Mediation Bootstrap Results (N={n_boot}) ---\n"
        report_str += "Indirect Effect: General_Grumbling -> Willingness -> Long_Term_Adoption\n"
        report_str += f"Parameter 1 ('a' path): {p_a}\n"
        report_str += f"Parameter 2 ('b' path): {p_b}\n\n"
        report_str += str(indirect_summary)
        
        _save_to_file(report_str, 'mediation_results.txt', output_dir)
        results_log['mediation_results'] = boot_results
        results_log['mediation_message'] = "Success: Bootstrap mediation complete."
        
    except Exception as e:
        print(f"   ! ERROR running mediation: {e}")
        results_log['mediation_results'] = None
        results_log['mediation_message'] = f"Error: {e}"

# --- FIX: Completed run_moderation function ---
def run_moderation(df_processed, output_dir, results_log):
    """
    3F. Runs Moderation analysis (using regression).
    """
    print(f"\n--- 3F. Moderation Analysis ---")
    
    # Example moderation:
    # Does Work_Overload moderate the effect of Switching_Cost on General_Grumbling?
    # DV: General_Grumbling
    # IV: Switching_Cost
    # MOD: Work_Overload
    
    df_mod = df_processed[['General_Grumbling', 'Switching_Cost', 'Work_Overload']].dropna()
    
    if df_mod.shape[0] < 50:
        print("   ! SKIPPING: Not enough data for moderation example.")
        results_log['moderation_results'] = None
        results_log['moderation_message'] = "Skipped: Insufficient data."
        return
        
    try:
        print("   Running moderation (Work_Overload on Switching_Cost -> General_Grumbling)")
        
        # Center variables to reduce multicollinearity
        df_mod['SC_C'] = df_mod['Switching_Cost'] - df_mod['Switching_Cost'].mean()
        df_mod['WO_C'] = df_mod['Work_Overload'] - df_mod['Work_Overload'].mean()
        
        # Create interaction term
        df_mod['Interaction'] = df_mod['SC_C'] * df_mod['WO_C']
        
        # OLS Regression formula
        formula = 'General_Grumbling ~ SC_C + WO_C + Interaction'
        
        # Run the model
        model = smf.ols(formula, data=df_mod).fit()
        
        report_str = f"--- OLS Moderation Results ---\n"
        report_str += f"Model: {formula}\n\n"
        report_str += str(model.summary())
        
        # Check for VIF (multicollinearity)
        X = df_mod[['SC_C', 'WO_C', 'Interaction']]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        report_str += "\n\n--- Variance Inflation Factor (VIF) ---\n"
        report_str += str(vif_data.to_string(index=False))
        
        _save_to_file(report_str, 'moderation_results.txt', output_dir)
        
        results_log['moderation_results'] = model
        results_log['moderation_message'] = "Success: Moderation analysis complete."
        
    except Exception as e:
        print(f"   ! ERROR running moderation: {e}")
        results_log['moderation_results'] = None
        results_log['moderation_message'] = f"Error: {e}"
# --- End of run_moderation function ---

# -----------------------------------------------------------------------------
# 4. Main Execution Function
# -----------------------------------------------------------------------------
def run_full_analysis(df_raw, column_rename_map, input_file, output_dir='analysis_outputs'):
    """
    Main function to execute the entire analysis pipeline.
    
    :param df_raw: The raw DataFrame loaded from the user's file.
    :param column_rename_map: Dictionary mapping raw column names to script names.
    :param input_file: Name of the input file (for logging purposes).
    :param output_dir: Directory to save all output files.
    :return: A dictionary containing all statistical results objects.
    """
    start_time = datetime.datetime.now()
    results_log = {
        'metadata': {
            'input_file': input_file,
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'python_version': sys.version.split('\n')[0],
            'pandas_version': pd.__version__
        }
    }
    
    # Setup environment
    _setup_environment(SEED, output_dir)
    _print_optional_library_warnings()
    
    # Apply user-defined column renaming
    print("\n--- Renaming Columns ---")
    df = df_raw.rename(columns=column_rename_map)
    print(f"Renamed {len(column_rename_map)} columns.")
    
    # 1. Preprocessing
    df_clean = preprocess_data(df, LIKERT_MAP, CONSTRUCT_MAP, output_dir)
    
    # 2. Create Constructs
    df_processed = create_constructs(df_clean, CONSTRUCT_MAP, output_dir)
    
    # Extract only the item columns for EFA/CFA
    df_items = df_clean[ALL_ITEM_NAMES]

    # 3. Statistical Analysis Steps
    
    # 3A. Descriptives & Reliability
    run_descriptives_and_reliability(df_processed, CONSTRUCT_MAP, output_dir, results_log)
    
    # 3B. EFA
    run_efa(df_items, output_dir, _FALLBACK_FLAGS['has_factor_analyzer'], results_log)
    
    # 3C. CFA
    run_cfa(df_items, CONSTRUCT_MAP, output_dir, _FALLBACK_FLAGS['has_semopy'], results_log)
    
    # 3D. SEM
    run_sem(df_processed, SEM_MODEL_SPEC, REGRESSION_FALLBACK_MODELS, 
            output_dir, _FALLBACK_FLAGS['has_semopy'], _FALLBACK_FLAGS['has_graphviz'], 
            results_log, SMALL_SAMPLE_THRESHOLD)
    
    # 3E. Mediation
    sem_model_obj = results_log.get('sem_model_obj', None)
    run_mediation(df_processed, sem_model_obj, output_dir, 
                  _FALLBACK_FLAGS['has_semopy'], BOOTSTRAP_SAMPLES, results_log)
    
    # 3F. Moderation (using OLS regression)
    run_moderation(df_processed, output_dir, results_log)
    
    # Final Summary
    end_time = datetime.datetime.now()
    results_log['metadata']['end_time'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    results_log['metadata']['duration'] = str(end_time - start_time)
    
    # Save a final text summary of the results log
    summary_content = _create_results_summary(results_log)
    _save_to_file(summary_content, 'analysis_log_summary.txt', output_dir)
    
    print("\n--- Pipeline Complete ---")
    print(f"Total Duration: {results_log['metadata']['duration']}")
    print(f"All outputs saved to: {os.path.abspath(output_dir)}")
    
    return results_log

# -----------------------------------------------------------------------------
# 5. Execution Block (How to run the script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- USER INPUT SECTION ---
    
    # 1. Define your input file and the output folder name
    INPUT_FILE = "./grumble.xlsx"  # <<< CHANGE THIS FILENAME
    OUTPUT_FOLDER = "analysis_outputs"
    
    # 2. Define the exact column renaming map from your raw data 
    #    to the script's internal names (from the previous steps).
    #    The left side MUST EXACTLY match the header in your file.
    #    The right side MUST EXACTLY match the required script name (e.g., 'sc1', 'cgpa').
    
    column_rename_map = {
        # 'Raw Column Name from your Excel file'       : 'script_name',
        
        # --- Demographic Columns ---
        'Your Current Average CGPA ( Approximate)'  : 'cgpa',
        'How many non-STEM courses have you completed?': 'non_stem_count',
        'What year are you currently in?'            : 'year',
        'You Gender'                                 : 'gender',
        
        # --- Likert Item Columns (Must match CONSTRUCT_MAP) ---
        'Adjusting my learning style for non-STEM courses is more challenging than for STEM courses. (Switching Cost)': 'sc1',
        'Switching focus from STEM to non-STEM subjects is mentally exhausting. (Switching Cost)': 'sc2',
        'Non-STEM courses improve my soft skills, such as communication and critical thinking. (Switching Benefit)': 'sb1',
        'Studying non-STEM subjects broadens my overall perspective on learning.': 'sb2',
        'Non-STEM courses significantly increase my academic workload. (Work Overload)': 'wo1',
        'Managing both STEM and non-STEM courses is challenging. (Work Overload)': 'wo2',
        'I am unsure how non-STEM courses contribute to my academic goals. (Role Ambiguity)': 'ra1',
        'I struggle to see the practical application of non-STEM subjects compared to STEM subjects. (Role Ambiguity)': 'ra2',
        'Balancing non-STEM and STEM coursework leaves me with less time for personal activities. (Work–Home Conflict)': 'whc1',
        'I feel that non-STEM courses are a waste of my time compared to STEM courses. (General Grumbling)': 'gg1',
        'Courses in the humanities (e.g., literature, philosophy) are unnecessary for STEM students.': 'gg2',
        'Social science courses (e.g., sociology, psychology) seem less relevant to my field than STEM subjects."': 'gg3',
        'I believe non-STEM courses are valuable despite their challenges. (Willingness)': 'w1',
        'I am open to integrating lessons from non-STEM courses into my academic work.  (Willingness)': 'w2',
        'I will likely apply the knowledge I gain from non-STEM courses in my personal or professional life. (Long-Term Adoption)': 'lta1'
    }
    
    # 3. Load the Data
    print(f"Attempting to load data from: {INPUT_FILE}")
    try:
        if INPUT_FILE.endswith('.csv'):
            df_raw = pd.read_csv(INPUT_FILE)
        elif INPUT_FILE.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(INPUT_FILE)
        else:
            print(f"FATAL ERROR: Unsupported file format. Please use .csv or .xlsx.", file=sys.stderr)
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at '{INPUT_FILE}'.", file=sys.stderr)
        print("Please ensure the file is in the same directory as the script or the path is correct.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Could not load data file. Reason: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Run the Analysis
    run_full_analysis(df_raw, column_rename_map, INPUT_FILE, OUTPUT_FOLDER)