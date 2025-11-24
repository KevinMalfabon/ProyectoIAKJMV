"""
Proyecto Final - Razonamiento con IA
Parte 3: Algoritmos de Clasificación Supervisada
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, mean_absolute_error,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_datos_hojas(ruta='hojas_dataset.csv'):
    """Carga el dataset de hojas"""
    df = pd.read_csv(ruta)
    X = df[['length_cm', 'width_cm']].values
    y = df['species_label'].values
    nombres_especies = ['Jacaranda', 'Abedul', 'Fresno']
    nombres_caracteristicas = ['length_cm', 'width_cm']
    return X, y, nombres_especies, nombres_caracteristicas

def cargar_datos_iris():
    """Carga el dataset Iris"""
    data = load_iris()
    X = data.data
    y = data.target
    nombres_especies = ['Setosa', 'Versicolor', 'Virginica']
    nombres_caracteristicas = data.feature_names
    return X, y, nombres_especies, nombres_caracteristicas

def evaluar_modelo_cv(modelo, X, y, cv=10):
    """Evalúa un modelo usando validación cruzada k-fold"""
    
    # Definir métricas
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    # Validación cruzada
    cv_results = cross_validate(modelo, X, y, cv=cv, scoring=scoring, 
                                return_train_score=False)
    
    # Calcular MAE manualmente
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    mae_scores = []
    confusion_matrices = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
    # Matriz de confusión promedio
    confusion_avg = np.mean(confusion_matrices, axis=0)
    
    resultados = {
        'accuracy': cv_results['test_accuracy'].mean(),
        'accuracy_std': cv_results['test_accuracy'].std(),
        'precision': cv_results['test_precision_macro'].mean(),
        'precision_std': cv_results['test_precision_macro'].std(),
        'recall': cv_results['test_recall_macro'].mean(),
        'recall_std': cv_results['test_recall_macro'].std(),
        'f1': cv_results['test_f1_macro'].mean(),
        'f1_std': cv_results['test_f1_macro'].std(),
        'mae': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'confusion_matrix': confusion_avg
    }
    
    return resultados

def imprimir_resultados(nombre_algoritmo, resultados, nombres_especies):
    """Imprime los resultados de evaluación de forma formateada"""
    
    print(f"\n{'='*80}")
    print(f"{nombre_algoritmo.upper()}")
    print(f"{'='*80}\n")
    
    print("MÉTRICAS DE DESEMPEÑO (10-fold Cross Validation):\n")
    print(f"  Accuracy:  {resultados['accuracy']:.4f} ± {resultados['accuracy_std']:.4f}")
    print(f"  Precision: {resultados['precision']:.4f} ± {resultados['precision_std']:.4f}")
    print(f"  Recall:    {resultados['recall']:.4f} ± {resultados['recall_std']:.4f}")
    print(f"  F1-Score:  {resultados['f1']:.4f} ± {resultados['f1_std']:.4f}")
    print(f"  MAE:       {resultados['mae']:.4f} ± {resultados['mae_std']:.4f}")
    
    print(f"\nMATRIZ DE CONFUSIÓN (promedio de 10 folds):\n")
    confusion_df = pd.DataFrame(
        resultados['confusion_matrix'],
        index=[f"Real {esp}" for esp in nombres_especies],
        columns=[f"Pred {esp}" for esp in nombres_especies]
    )
    print(confusion_df.round(1))

# ============================================================================
# K-NEAREST NEIGHBORS (k-NN)
# ============================================================================

def entrenar_knn(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa k-NN con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# K-NEAREST NEIGHBORS (k-NN) - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X_scaled, y, cv=10)
    
    imprimir_resultados("K-Nearest Neighbors", resultados, nombres_especies)
    
    return {
        'nombre': 'k-NN',
        'modelo': mejor_modelo,
        'scaler': scaler,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# NAIVE BAYES
# ============================================================================

def entrenar_naive_bayes(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa Naive Bayes con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# NAIVE BAYES - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'var_smoothing': np.logspace(-10, -5, 20)
    }
    
    nb = GaussianNB()
    grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value:.2e}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X_scaled, y, cv=10)
    
    imprimir_resultados("Naive Bayes", resultados, nombres_especies)
    
    return {
        'nombre': 'Naive Bayes',
        'modelo': mejor_modelo,
        'scaler': scaler,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# SUPPORT VECTOR MACHINES (SVM)
# ============================================================================

def entrenar_svm(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa SVM con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# SUPPORT VECTOR MACHINES (SVM) - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X_scaled, y, cv=10)
    
    imprimir_resultados("Support Vector Machines", resultados, nombres_especies)
    
    # Información adicional para SVM
    mejor_modelo.fit(X_scaled, y)
    print(f"\nNúmero de vectores de soporte: {len(mejor_modelo.support_)}")
    print(f"Porcentaje de vectores de soporte: {len(mejor_modelo.support_)/len(X)*100:.2f}%")
    
    return {
        'nombre': 'SVM',
        'modelo': mejor_modelo,
        'scaler': scaler,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# ÁRBOL DE DECISIÓN
# ============================================================================

def entrenar_arbol(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa Árbol de Decisión con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# ÁRBOL DE DECISIÓN - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    arbol = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(arbol, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X, y, cv=10)
    
    imprimir_resultados("Árbol de Decisión", resultados, nombres_especies)
    
    # Información adicional
    mejor_modelo.fit(X, y)
    print(f"\nProfundidad del árbol: {mejor_modelo.get_depth()}")
    print(f"Número de hojas: {mejor_modelo.get_n_leaves()}")
    
    return {
        'nombre': 'Árbol de Decisión',
        'modelo': mejor_modelo,
        'scaler': None,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# RANDOM FOREST
# ============================================================================

def entrenar_random_forest(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa Random Forest con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# RANDOM FOREST - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X, y, cv=10)
    
    imprimir_resultados("Random Forest", resultados, nombres_especies)
    
    # Información adicional
    mejor_modelo.fit(X, y)
    print(f"\nNúmero de árboles: {mejor_modelo.n_estimators}")
    if hasattr(mejor_modelo, 'oob_score_'):
        print(f"OOB Score: {mejor_modelo.oob_score_:.4f}")
    
    return {
        'nombre': 'Random Forest',
        'modelo': mejor_modelo,
        'scaler': None,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# MULTILAYER PERCEPTRON (MLP)
# ============================================================================

def entrenar_mlp(X, y, nombres_especies, nombre_dataset):
    """Entrena y evalúa MLP con búsqueda de hiperparámetros"""
    
    print(f"\n{'#'*80}")
    print(f"# MULTILAYER PERCEPTRON (MLP) - {nombre_dataset}")
    print(f"{'#'*80}")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Búsqueda de hiperparámetros
    print("\nBuscando mejores hiperparámetros...")
    
    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [1000, 2000]
    }
    
    mlp = MLPClassifier(random_state=42, early_stopping=True)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    print(f"\nMejores hiperparámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor F1-Score en búsqueda: {grid_search.best_score_:.4f}")
    
    # Evaluar con validación cruzada
    mejor_modelo = grid_search.best_estimator_
    resultados = evaluar_modelo_cv(mejor_modelo, X_scaled, y, cv=10)
    
    imprimir_resultados("Multilayer Perceptron", resultados, nombres_especies)
    
    # Información adicional
    mejor_modelo.fit(X_scaled, y)
    print(f"\nNúmero de iteraciones: {mejor_modelo.n_iter_}")
    print(f"Loss final: {mejor_modelo.loss_:.4f}")
    
    return {
        'nombre': 'MLP',
        'modelo': mejor_modelo,
        'scaler': scaler,
        'mejores_params': grid_search.best_params_,
        'resultados': resultados
    }

# ============================================================================
# COMPARACIÓN DE TODOS LOS ALGORITMOS
# ============================================================================

def comparar_algoritmos(resultados_lista, nombre_dataset):
    """Compara todos los algoritmos y genera ranking"""
    
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN DE ALGORITMOS - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    # Crear DataFrame con resultados
    datos_comparacion = []
    for res in resultados_lista:
        datos_comparacion.append({
            'Algoritmo': res['nombre'],
            'Accuracy': res['resultados']['accuracy'],
            'Precision': res['resultados']['precision'],
            'Recall': res['resultados']['recall'],
            'F1-Score': res['resultados']['f1'],
            'MAE': res['resultados']['mae'],
            'F1_std': res['resultados']['f1_std']
        })
    
    df_comparacion = pd.DataFrame(datos_comparacion)
    df_comparacion = df_comparacion.sort_values('F1-Score', ascending=False)
    df_comparacion['Ranking'] = range(1, len(df_comparacion) + 1)
    
    # Reordenar columnas
    df_comparacion = df_comparacion[['Ranking', 'Algoritmo', 'F1-Score', 'F1_std', 
                                     'Accuracy', 'Precision', 'Recall', 'MAE']]
    
    print("RANKING POR F1-SCORE:\n")
    print(df_comparacion.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    return df_comparacion

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el análisis de clasificación supervisada"""
    
    print("\n" + "="*80)
    print("PROYECTO FINAL - RAZONAMIENTO CON IA")
    print("PARTE 3: ALGORITMOS DE CLASIFICACIÓN SUPERVISADA")
    print("="*80)
    
    # ========================================================================
    # CLASIFICACIÓN EN LA BASE DE HOJAS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# CLASIFICACIÓN EN LA BASE DE DATOS DE HOJAS")
    print("#"*80)
    
    X_hojas, y_hojas, nombres_hojas, _ = cargar_datos_hojas()
    
    resultados_hojas = []
    
    # k-NN
    resultados_hojas.append(entrenar_knn(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # Naive Bayes
    resultados_hojas.append(entrenar_naive_bayes(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # SVM
    resultados_hojas.append(entrenar_svm(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # Árbol de Decisión
    resultados_hojas.append(entrenar_arbol(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # Random Forest
    resultados_hojas.append(entrenar_random_forest(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # MLP
    resultados_hojas.append(entrenar_mlp(X_hojas, y_hojas, nombres_hojas, "Base de Hojas"))
    
    # Comparación
    comparacion_hojas = comparar_algoritmos(resultados_hojas, "Base de Hojas")
    
    # ========================================================================
    # CLASIFICACIÓN EN LA BASE DE IRIS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# CLASIFICACIÓN EN LA BASE DE DATOS IRIS")
    print("#"*80)
    
    X_iris, y_iris, nombres_iris, _ = cargar_datos_iris()
    
    resultados_iris = []
    
    # k-NN
    resultados_iris.append(entrenar_knn(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # Naive Bayes
    resultados_iris.append(entrenar_naive_bayes(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # SVM
    resultados_iris.append(entrenar_svm(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # Árbol de Decisión
    resultados_iris.append(entrenar_arbol(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # Random Forest
    resultados_iris.append(entrenar_random_forest(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # MLP
    resultados_iris.append(entrenar_mlp(X_iris, y_iris, nombres_iris, "Iris Dataset"))
    
    # Comparación
    comparacion_iris = comparar_algoritmos(resultados_iris, "Iris Dataset")
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("GUARDANDO RESULTADOS...")
    print("="*80)
    
    comparacion_hojas.to_csv('ranking_clasificacion_hojas.csv', index=False)
    comparacion_iris.to_csv('ranking_clasificacion_iris.csv', index=False)
    
    # Guardar modelos entrenados
    import pickle
    with open('modelos_hojas.pkl', 'wb') as f:
        pickle.dump(resultados_hojas, f)
    with open('modelos_iris.pkl', 'wb') as f:
        pickle.dump(resultados_iris, f)
    
    print("\n✓ Rankings guardados en archivos CSV")
    print("✓ Modelos guardados en archivos PKL")
    print("\n" + "="*80)
    print("ANÁLISIS DE CLASIFICACIÓN SUPERVISADA COMPLETADO")
    print("="*80 + "\n")
    
    return resultados_hojas, resultados_iris

if __name__ == "__main__":
    resultados_hojas, resultados_iris = main()
