import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score

# Traducir de probabilidad de default a G.
def proba_to_score(proba: float, limits: dict) -> str:
    for idx, score_range in enumerate(limits.items()):
        if idx == 0:
            if proba >= score_range[1][0] and proba <= score_range[1][1]:
                return score_range[0]
        else:
            if proba > score_range[1][0] and proba <= score_range[1][1]:
                return score_range[0]

    return None

# Convertir el vector completo de probabilidades de default a Gs.
def get_scores(y_proba: np.array, limits: dict) -> np.ndarray:
    return np.array([proba_to_score(p, limits) for p in y_proba])

# Métricas sobre la base agregada a nivel de G.
def groups_stats(df_groups: pd.DataFrame, limits: dict):
    states = []
    midpoint_distances = []
    distances = []

    for score, default_rate in zip(df_groups['score'], df_groups['default_rate']):
        ranges = limits[score]

        if ranges[0] == 0: # Verifica si el límite inferior es 0 (G1)       
            if default_rate <= ranges[1]: 
                states.append('en_rango')
                distances.append(0)
            else: 
                states.append('subestimacion')
                distances.append(default_rate - ranges[1])
        else:
            if default_rate <= ranges[0]: 
                states.append('sobreestimacion')
                distances.append(ranges[0] - default_rate)
            elif default_rate <= ranges[1]: 
                states.append('en_rango')
                distances.append(0)
            else: 
                states.append('subestimacion')
                distances.append(default_rate - ranges[1])
            
        midpoint = abs(((ranges[1] - ranges[0]) / 2) - ranges[1])
        midpoint_distances.append(abs(default_rate - midpoint))

    return states, np.array(midpoint_distances), distances

def limits_gains(df_gains: pd.DataFrame, limits: dict):
    limits_table = [(row['score'], limits[row['score']][0], limits[row['score']][1]) for idx, row in df_gains.iterrows()]
    return pd.DataFrame(limits_table, columns=['score', 'lower', 'upper'])

# Tabla agregada a nivel de G con los resultados del modelo.
def cumulative_gains_table(y_true: np.ndarray, y_proba: np.ndarray, limits: dict, percentage: bool) -> pd.DataFrame:
    scores = get_scores(y_proba, limits)

    df_groups = pd.DataFrame(zip(scores, y_true), columns=['score', 'default'])
    df_groups = df_groups.groupby('score').agg(default=('default', 'sum'), customers=('default', 'count')).reset_index()
    
    df_groups = df_groups.assign(default_rate=df_groups['default'] / df_groups['customers'])
    df_groups = df_groups.assign(pct_customers=df_groups['customers'] / sum(df_groups['customers']))    
    df_groups = df_groups.assign(pct_customers_cum=df_groups['pct_customers'].cumsum())
    
    states, midpoint_distances, distances = groups_stats(df_groups, limits)
    df_groups = df_groups.assign(state=states)
    df_groups = df_groups.assign(mp_distance=midpoint_distances)
    df_groups = df_groups.assign(distance=distances)
    
    limits_table = limits_gains(df_groups, limits)
    df_groups = df_groups.merge(limits_table, on='score')
    
    if percentage:
        df_groups = df_groups.assign(default_rate=round(df_groups['default_rate'] * 100, 2))
        df_groups = df_groups.assign(pct_customers=round(df_groups['pct_customers'] * 100, 2))
        df_groups = df_groups.assign(pct_customers_cum=round(df_groups['pct_customers_cum'] * 100, 2))
        df_groups = df_groups.assign(lower=round(df_groups['lower'] * 100, 2))
        df_groups = df_groups.assign(upper=round(df_groups['upper'] * 100, 2))
        
    return df_groups[['score', 'default', 'customers', 'default_rate', 'lower', 'upper', 'state', 'pct_customers', 'pct_customers_cum', 'mp_distance', 'distance']]

# Verificar si la TDO se encuentra en rango.
def is_ordered(df_groups: pd.DataFrame) -> int:
    diff = df_groups['default_rate'].diff().dropna().to_numpy()
    return int(0 not in np.where(diff < 0, 0, 1))

# Métricas básica de la iteración.
def basic_metrics(df_groups: pd.DataFrame, set_name: str) -> dict:
    metrics = {}
    
    metrics['n_escalas_' + set_name] = len(df_groups.index)
    metrics['n_en_rango_' + set_name] = len(df_groups[df_groups['state'] == 'en_rango'].index)
    metrics['n_subestimadas_' + set_name] = len(df_groups[df_groups['state'] == 'subestimacion'].index)
    metrics['n_sobreestimadas_' + set_name] = len(df_groups[df_groups['state'] == 'sobreestimacion'].index)
    metrics['mp_dist_' + set_name] = sum(df_groups['mp_distance'])
    metrics['dist_' + set_name] = sum(df_groups['distance'])
    metrics['ordenado_' + set_name] = is_ordered(df_groups)

    return metrics

def get_groups_by_range(range_item: tuple) -> list:
    groups = ['G' + str(r) for r in range(int(range_item[0][1]), int(range_item[1][1]) + 1)]
    return groups  

def get_subset_name(set_name: str, range_item: tuple) -> str:
    return str(range_item[0]).lower() + '_' + str(range_item[1]).lower() + '_' + set_name

def get_metrics(y_true: np.ndarray, y_proba: np.ndarray, ranges: list, limits: dict, set_name: str) -> dict:
    df_gains = cumulative_gains_table(y_true, y_proba, limits, percentage=False)
    
    metrics = basic_metrics(df_gains, set_name)
    metrics['roc_auc_' + set_name] = roc_auc_score(y_true, y_proba)
  
    for range_item in ranges:
        subset_name = get_subset_name(set_name, range_item)
        df_sub = df_gains[df_gains['score'].isin(get_groups_by_range(range_item))]

        metrics.update(basic_metrics(df_sub, subset_name))

    return metrics

# Comparar importancia de dos variables.
def compare_importances(row, importances: dict) -> str:
    if importances[row['feature_1']] > importances[row['feature_2']]: 
        return row['feature_1']
    elif importances[row['feature_2']] > importances[row['feature_1']]: 
        return row['feature_2']
    else:
        return 'Error'

# Filtrar variables correlacionadas.
def filter_corr(X: pd.DataFrame, features: list, importances: dict, threshold: float):
    unnecessary_features = set()

    # Correlation matrix.
    corr_df = X[features].corr()

    # Upper triangular matrix.
    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(bool))
    corr_df = corr_df.stack().reset_index()
    corr_df.columns = ['feature_1', 'feature_2', 'corr']
    df_corr_copy = corr_df.copy()

    # Ignore main diagonal of matrix.
    corr_df = corr_df[corr_df['feature_1'] != corr_df['feature_2']]

    # Select all features that exceed the threshold
    corr_df = corr_df[corr_df['corr'] >= threshold]

    stop = False    
    while not stop:
        dispute_features = set(corr_df['feature_1']).union(set(corr_df['feature_2']))

        winning_features = {compare_importances(row, importances) for idx, row in corr_df.iterrows()}
        losing_features = dispute_features - winning_features

        corr_df = corr_df[corr_df['feature_1'].isin(winning_features)]
        corr_df = corr_df[corr_df['feature_2'].isin(winning_features)]

        if len(losing_features) == 0: stop = True
        else: unnecessary_features = unnecessary_features.union(losing_features)
                
    selected_features = list(set(features) - unnecessary_features)

    return selected_features, df_corr_copy

# Obtener tasa de default de un DataFrame.
def get_default_rate(data: pd.DataFrame) -> float:
    default_counter = Counter(data['default'])
    return default_counter[1] / (default_counter[1] + default_counter[0])

# Contar variables por módulo de información.
def count_features_modules(features):
    counts = defaultdict(int)
    for f in features:
        module = f.split('_')[0]
        
        if f in ['nivel_academico', 'ocupacion', 'estado_civil']:
            counts['demograficas'] += 1
        else:
            counts[module] += 1
            
    df_counts = pd.DataFrame(counts.items(), columns=['module', 'n_features'])
    df_counts = df_counts.sort_values(by='n_features', ascending=False)
    df_counts['%'] = df_counts['n_features'] / sum(df_counts['n_features'])
    
    return df_counts