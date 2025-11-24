"""
Proyecto Final - Razonamiento con IA
Parte 4: Visualizaciones de Fronteras de Decisión
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pickle
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
    return X, y, nombres_especies

def cargar_datos_iris():
    """Carga el dataset Iris (solo 2 características para visualización)"""
    data = load_iris()
    # Usar petal length y petal width (más discriminativas)
    X = data.data[:, 2:4]
    y = data.target
    nombres_especies = ['Setosa', 'Versicolor', 'Virginica']
    return X, y, nombres_especies

def crear_malla(X, h=0.02):
    """Crea una malla de puntos para visualizar fronteras de decisión"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# ============================================================================
# VISUALIZACIÓN DE FRONTERAS DE DECISIÓN
# ============================================================================

def plot_decision_boundary(modelo, X, y, nombres_especies, titulo, scaler=None, ax=None):
    """Grafica la frontera de decisión de un modelo"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear malla
    xx, yy = crear_malla(X, h=0.02)
    
    # Predecir en la malla
    Z_input = np.c_[xx.ravel(), yy.ravel()]
    if scaler is not None:
        Z_input = scaler.transform(Z_input)
    
    Z = modelo.predict(Z_input)
    Z = Z.reshape(xx.shape)
    
    # Colores
    colores = ['#FFB6C1', '#B0E0E6', '#98FB98']
    colores_puntos = ['#FF1493', '#1E90FF', '#32CD32']
    marcadores = ['o', 's', '^']
    
    # Graficar frontera
    ax.contourf(xx, yy, Z, alpha=0.4, levels=np.arange(len(nombres_especies)+1)-0.5,
               colors=colores)
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, 
              levels=np.arange(len(nombres_especies)+1)-0.5)
    
    # Graficar puntos
    for clase_id, nombre_clase in enumerate(nombres_especies):
        mask = y == clase_id
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=colores_puntos[clase_id], 
                  marker=marcadores[clase_id],
                  s=100, 
                  alpha=0.8, 
                  edgecolors='black',
                  linewidth=1.5,
                  label=nombre_clase)
    
    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

def visualizar_todas_fronteras(X, y, nombres_especies, resultados_lista, nombre_dataset):
    """Visualiza las fronteras de decisión de todos los algoritmos"""
    
    print(f"\nGenerando visualizaciones de fronteras de decisión para {nombre_dataset}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()
    
    for idx, res in enumerate(resultados_lista):
        modelo = res['modelo']
        scaler = res['scaler']
        nombre = res['nombre']
        
        plot_decision_boundary(modelo, X, y, nombres_especies, 
                             nombre, scaler, ax=axes[idx])
    
    plt.suptitle(f'Fronteras de Decisión - {nombre_dataset}', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'fronteras_decision_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualización guardada")

# ============================================================================
# VISUALIZACIÓN INDIVIDUAL DE CADA ALGORITMO
# ============================================================================

def visualizar_fronteras_individuales(X, y, nombres_especies, resultados_lista, nombre_dataset):
    """Genera visualizaciones individuales de alta calidad para cada algoritmo"""
    
    print(f"\nGenerando visualizaciones individuales para {nombre_dataset}...")
    
    for res in resultados_lista:
        modelo = res['modelo']
        scaler = res['scaler']
        nombre = res['nombre']
        
        fig, ax = plt.subplots(figsize=(12, 9))
        plot_decision_boundary(modelo, X, y, nombres_especies, 
                             f'{nombre} - {nombre_dataset}', scaler, ax=ax)
        
        plt.tight_layout()
        nombre_archivo = f'frontera_{nombre.lower().replace(" ", "_")}_{nombre_dataset.lower().replace(" ", "_")}.png'
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {nombre}")
    
    print(f"✓ Todas las visualizaciones individuales guardadas")

# ============================================================================
# VISUALIZACIÓN DE MATRICES DE CONFUSIÓN
# ============================================================================

def visualizar_matrices_confusion(resultados_lista, nombres_especies, nombre_dataset):
    """Visualiza las matrices de confusión de todos los algoritmos"""
    
    print(f"\nGenerando matrices de confusión para {nombre_dataset}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, res in enumerate(resultados_lista):
        nombre = res['nombre']
        confusion = res['resultados']['confusion_matrix']
        
        sns.heatmap(confusion, annot=True, fmt='.1f', cmap='Blues', ax=axes[idx],
                   xticklabels=nombres_especies,
                   yticklabels=nombres_especies,
                   cbar_kws={'label': 'Count'})
        
        axes[idx].set_title(f'{nombre}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Clase Real', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Clase Predicha', fontsize=11, fontweight='bold')
    
    plt.suptitle(f'Matrices de Confusión - {nombre_dataset}', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'matrices_confusion_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Matrices de confusión guardadas")

# ============================================================================
# VISUALIZACIÓN COMPARATIVA DE MÉTRICAS
# ============================================================================

def visualizar_comparacion_metricas(resultados_lista, nombre_dataset):
    """Visualiza la comparación de métricas entre algoritmos"""
    
    print(f"\nGenerando comparación de métricas para {nombre_dataset}...")
    
    # Extraer datos
    nombres = [res['nombre'] for res in resultados_lista]
    f1_scores = [res['resultados']['f1'] for res in resultados_lista]
    f1_stds = [res['resultados']['f1_std'] for res in resultados_lista]
    accuracies = [res['resultados']['accuracy'] for res in resultados_lista]
    precisions = [res['resultados']['precision'] for res in resultados_lista]
    recalls = [res['resultados']['recall'] for res in resultados_lista]
    maes = [res['resultados']['mae'] for res in resultados_lista]
    
    # Ordenar por F1-score
    indices = np.argsort(f1_scores)[::-1]
    nombres = [nombres[i] for i in indices]
    f1_scores = [f1_scores[i] for i in indices]
    f1_stds = [f1_stds[i] for i in indices]
    accuracies = [accuracies[i] for i in indices]
    precisions = [precisions[i] for i in indices]
    recalls = [recalls[i] for i in indices]
    maes = [maes[i] for i in indices]
    
    # Crear visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colores = plt.cm.viridis(np.linspace(0.2, 0.9, len(nombres)))
    
    # 1. F1-Score con barras de error
    ax1 = axes[0, 0]
    bars = ax1.barh(nombres, f1_scores, xerr=f1_stds, color=colores, 
                    edgecolor='black', linewidth=1.5, capsize=5)
    ax1.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score por Algoritmo', fontsize=14, fontweight='bold')
    ax1.set_xlim([min(f1_scores) - 0.02, 1.0])
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 2. Comparación de todas las métricas
    ax2 = axes[0, 1]
    x = np.arange(len(nombres))
    width = 0.2
    
    ax2.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#FF6B6B', 
           edgecolor='black', linewidth=1)
    ax2.bar(x - 0.5*width, precisions, width, label='Precision', color='#4ECDC4', 
           edgecolor='black', linewidth=1)
    ax2.bar(x + 0.5*width, recalls, width, label='Recall', color='#45B7D1', 
           edgecolor='black', linewidth=1)
    ax2.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#FFA07A', 
           edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Métricas', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.set_ylim([min(min(accuracies), min(precisions), min(recalls), min(f1_scores)) - 0.02, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. MAE
    ax3 = axes[1, 0]
    bars = ax3.barh(nombres, maes, color=colores, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax3.set_title('MAE por Algoritmo (menor es mejor)', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_xaxis()  # Invertir para que menor esté a la derecha
    
    for i, (bar, mae) in enumerate(zip(bars, maes)):
        width = bar.get_width()
        ax3.text(width - 0.002, bar.get_y() + bar.get_height()/2, 
                f'{mae:.4f}', ha='right', va='center', fontweight='bold', fontsize=10)
    
    # 4. Diagrama de radar
    ax4 = axes[1, 1]
    ax4.remove()
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    
    categorias = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    num_vars = len(categorias)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, nombre in enumerate(nombres):
        valores = [accuracies[i], precisions[i], recalls[i], f1_scores[i]]
        valores += valores[:1]
        
        ax4.plot(angles, valores, 'o-', linewidth=2, label=nombre, color=colores[i])
        ax4.fill(angles, valores, alpha=0.15, color=colores[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categorias, fontsize=10)
    ax4.set_ylim([0.9, 1.0])
    ax4.set_title('Diagrama de Radar - Métricas', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax4.grid(True)
    
    plt.suptitle(f'Comparación de Desempeño - {nombre_dataset}', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'comparacion_metricas_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Comparación de métricas guardada")

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que genera todas las visualizaciones"""
    
    print("\n" + "="*80)
    print("PROYECTO FINAL - RAZONAMIENTO CON IA")
    print("PARTE 4: VISUALIZACIONES DE FRONTERAS DE DECISIÓN Y MÉTRICAS")
    print("="*80)
    
    # Cargar modelos entrenados
    print("\nCargando modelos entrenados...")
    try:
        with open('modelos_hojas.pkl', 'rb') as f:
            resultados_hojas = pickle.load(f)
        with open('modelos_iris.pkl', 'rb') as f:
            resultados_iris = pickle.load(f)
        print("✓ Modelos cargados exitosamente")
    except FileNotFoundError:
        print("ERROR: No se encontraron los modelos entrenados.")
        print("Por favor, ejecuta primero '3_clasificacion_supervisada.py'")
        return
    
    # ========================================================================
    # VISUALIZACIONES PARA BASE DE HOJAS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# VISUALIZACIONES PARA BASE DE HOJAS")
    print("#"*80)
    
    X_hojas, y_hojas, nombres_hojas = cargar_datos_hojas()
    
    visualizar_todas_fronteras(X_hojas, y_hojas, nombres_hojas, 
                              resultados_hojas, "Base de Hojas")
    
    visualizar_fronteras_individuales(X_hojas, y_hojas, nombres_hojas, 
                                     resultados_hojas, "Base de Hojas")
    
    visualizar_matrices_confusion(resultados_hojas, nombres_hojas, "Base de Hojas")
    
    visualizar_comparacion_metricas(resultados_hojas, "Base de Hojas")
    
    # ========================================================================
    # VISUALIZACIONES PARA IRIS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# VISUALIZACIONES PARA IRIS DATASET")
    print("#"*80)
    
    X_iris, y_iris, nombres_iris = cargar_datos_iris()
    
    visualizar_todas_fronteras(X_iris, y_iris, nombres_iris, 
                              resultados_iris, "Iris Dataset")
    
    visualizar_fronteras_individuales(X_iris, y_iris, nombres_iris, 
                                     resultados_iris, "Iris Dataset")
    
    visualizar_matrices_confusion(resultados_iris, nombres_iris, "Iris Dataset")
    
    visualizar_comparacion_metricas(resultados_iris, "Iris Dataset")
    
    # ========================================================================
    # FINALIZACIÓN
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("TODAS LAS VISUALIZACIONES COMPLETADAS")
    print("="*80)
    print("\n✓ Fronteras de decisión generadas")
    print("✓ Matrices de confusión generadas")
    print("✓ Comparaciones de métricas generadas")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
