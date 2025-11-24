"""
Proyecto Final - Razonamiento con IA
Parte 1: Exploración y Análisis Descriptivo de Datos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def cargar_datos_hojas(ruta='hojas_dataset.csv'):
    """Carga el dataset de hojas desde CSV"""
    df = pd.read_csv(ruta)
    X = df[['length_cm', 'width_cm']].values
    y = df['species_label'].values
    nombres_especies = df['species_name'].unique()
    nombres_caracteristicas = ['length_cm', 'width_cm']
    
    return X, y, nombres_especies, nombres_caracteristicas, df

def cargar_datos_iris():
    """Carga el dataset Iris desde sklearn"""
    data = load_iris()
    X = data.data
    y = data.target
    nombres_especies = data.target_names
    nombres_caracteristicas = data.feature_names
    
    # Crear DataFrame
    df = pd.DataFrame(X, columns=nombres_caracteristicas)
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    return X, y, nombres_especies, nombres_caracteristicas, df

# ============================================================================
# ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

def estadisticas_globales(X, nombres_caracteristicas, nombre_dataset):
    """Calcula y muestra estadísticas descriptivas globales"""
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS GLOBALES - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    df_stats = pd.DataFrame(X, columns=nombres_caracteristicas)
    
    stats_dict = {
        'Variable': nombres_caracteristicas,
        'Media': df_stats.mean().values,
        'Desv. Estándar': df_stats.std().values,
        'Mínimo': df_stats.min().values,
        'Máximo': df_stats.max().values,
        'Rango': (df_stats.max() - df_stats.min()).values
    }
    
    tabla_stats = pd.DataFrame(stats_dict)
    print(tabla_stats.to_string(index=False))
    print(f"\nNúmero total de instancias: {len(X)}")
    
    return tabla_stats

def estadisticas_por_clase(X, y, nombres_caracteristicas, nombres_especies, nombre_dataset):
    """Calcula y muestra estadísticas descriptivas por clase"""
    print(f"\n{'='*80}")
    print(f"ESTADÍSTICAS POR CLASE - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(X, columns=nombres_caracteristicas)
    df['clase'] = y
    
    resultados = []
    
    for clase_id, nombre_clase in enumerate(nombres_especies):
        print(f"\n--- {nombre_clase.upper()} (Clase {clase_id}) ---")
        datos_clase = df[df['clase'] == clase_id][nombres_caracteristicas]
        
        print(f"Número de instancias: {len(datos_clase)}")
        print(f"\nEstadísticas:")
        
        for caracteristica in nombres_caracteristicas:
            valores = datos_clase[caracteristica]
            print(f"\n  {caracteristica}:")
            print(f"    Media: {valores.mean():.2f}")
            print(f"    Desv. Estándar: {valores.std():.2f}")
            print(f"    Mínimo: {valores.min():.2f}")
            print(f"    Máximo: {valores.max():.2f}")
            
            resultados.append({
                'Clase': nombre_clase,
                'Variable': caracteristica,
                'Media': valores.mean(),
                'Desv. Estándar': valores.std(),
                'Mínimo': valores.min(),
                'Máximo': valores.max()
            })
    
    return pd.DataFrame(resultados)

def matriz_correlacion(X, nombres_caracteristicas, nombre_dataset):
    """Calcula y visualiza la matriz de correlación"""
    df = pd.DataFrame(X, columns=nombres_caracteristicas)
    corr_matrix = df.corr()
    
    print(f"\n{'='*80}")
    print(f"MATRIZ DE CORRELACIÓN - {nombre_dataset}")
    print(f"{'='*80}\n")
    print(corr_matrix)
    
    # Visualización
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.3f')
    plt.title(f'Matriz de Correlación - {nombre_dataset}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'correlacion_{nombre_dataset.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

# ============================================================================
# VISUALIZACIONES EXPLORATORIAS
# ============================================================================

def grafico_dispersion_2d(X, y, nombres_especies, nombre_dataset, nombres_ejes=['Feature 1', 'Feature 2']):
    """Gráfico de dispersión 2D para las primeras dos características"""
    plt.figure(figsize=(12, 8))
    
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    marcadores = ['o', 's', '^']
    
    for clase_id, nombre_clase in enumerate(nombres_especies):
        mask = y == clase_id
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colores[clase_id], 
                   marker=marcadores[clase_id],
                   s=100, 
                   alpha=0.7, 
                   edgecolors='black',
                   linewidth=1,
                   label=nombre_clase)
    
    plt.xlabel(nombres_ejes[0], fontsize=14, fontweight='bold')
    plt.ylabel(nombres_ejes[1], fontsize=14, fontweight='bold')
    plt.title(f'Diagrama de Dispersión - {nombre_dataset}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'dispersion_{nombre_dataset.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def boxplots_por_clase(df, nombres_caracteristicas, nombre_dataset):
    """Boxplots comparativos por característica y clase"""
    n_caracteristicas = len(nombres_caracteristicas)
    fig, axes = plt.subplots(1, n_caracteristicas, figsize=(6*n_caracteristicas, 6))
    
    if n_caracteristicas == 1:
        axes = [axes]
    
    for idx, caracteristica in enumerate(nombres_caracteristicas):
        if 'species_name' in df.columns:
            columna_clase = 'species_name'
        else:
            columna_clase = 'species'
            
        sns.boxplot(data=df, x=columna_clase, y=caracteristica, 
                   palette='Set2', ax=axes[idx])
        axes[idx].set_title(f'{caracteristica}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Especie', fontsize=12)
        axes[idx].set_ylabel('Valor (cm)', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Distribución de Características por Especie - {nombre_dataset}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'boxplots_{nombre_dataset.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def pairplot_iris(df):
    """Pairplot para el dataset Iris (4 dimensiones)"""
    plt.figure(figsize=(14, 12))
    pairplot = sns.pairplot(df, hue='species_name', 
                            palette='husl',
                            markers=['o', 's', '^'],
                            diag_kind='hist',
                            plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'black', 'linewidth': 0.5},
                            diag_kws={'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 1})
    pairplot.fig.suptitle('Matriz de Diagramas de Dispersión - Iris Dataset', 
                          fontsize=16, fontweight='bold', y=1.01)
    plt.savefig('pairplot_iris.png', dpi=300, bbox_inches='tight')
    plt.show()

def histogramas_distribucion(df, nombres_caracteristicas, nombre_dataset):
    """Histogramas de distribución por característica"""
    n_caracteristicas = len(nombres_caracteristicas)
    fig, axes = plt.subplots(2, (n_caracteristicas + 1) // 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, caracteristica in enumerate(nombres_caracteristicas):
        axes[idx].hist(df[caracteristica], bins=20, color='skyblue', 
                      edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{caracteristica}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Valor', fontsize=10)
        axes[idx].set_ylabel('Frecuencia', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    # Ocultar ejes sobrantes
    for idx in range(n_caracteristicas, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Distribución de Características - {nombre_dataset}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'histogramas_{nombre_dataset.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el análisis exploratorio"""
    
    print("\n" + "="*80)
    print("PROYECTO FINAL - RAZONAMIENTO CON IA")
    print("PARTE 1: EXPLORACIÓN Y ANÁLISIS DESCRIPTIVO DE DATOS")
    print("="*80)
    
    # ========================================================================
    # ANÁLISIS DE LA BASE DE HOJAS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# ANÁLISIS DE LA BASE DE DATOS DE HOJAS")
    print("#"*80)
    
    X_hojas, y_hojas, nombres_hojas, caracteristicas_hojas, df_hojas = cargar_datos_hojas()
    
    # Estadísticas
    stats_globales_hojas = estadisticas_globales(X_hojas, caracteristicas_hojas, "Base de Hojas")
    stats_clase_hojas = estadisticas_por_clase(X_hojas, y_hojas, caracteristicas_hojas, 
                                                nombres_hojas, "Base de Hojas")
    
    # Correlación
    corr_hojas = matriz_correlacion(X_hojas, caracteristicas_hojas, "Base de Hojas")
    
    # Visualizaciones
    grafico_dispersion_2d(X_hojas, y_hojas, nombres_hojas, "Base de Hojas", 
                         nombres_ejes=['Longitud (cm)', 'Anchura (cm)'])
    boxplots_por_clase(df_hojas, caracteristicas_hojas, "Base de Hojas")
    histogramas_distribucion(df_hojas, caracteristicas_hojas, "Base de Hojas")
    
    # ========================================================================
    # ANÁLISIS DE LA BASE DE IRIS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# ANÁLISIS DE LA BASE DE DATOS IRIS")
    print("#"*80)
    
    X_iris, y_iris, nombres_iris, caracteristicas_iris, df_iris = cargar_datos_iris()
    
    # Estadísticas
    stats_globales_iris = estadisticas_globales(X_iris, caracteristicas_iris, "Iris Dataset")
    stats_clase_iris = estadisticas_por_clase(X_iris, y_iris, caracteristicas_iris, 
                                               nombres_iris, "Iris Dataset")
    
    # Correlación
    corr_iris = matriz_correlacion(X_iris, caracteristicas_iris, "Iris Dataset")
    
    # Visualizaciones
    grafico_dispersion_2d(X_iris, y_iris, nombres_iris, "Iris Dataset", 
                         nombres_ejes=['Sepal Length (cm)', 'Sepal Width (cm)'])
    boxplots_por_clase(df_iris, caracteristicas_iris, "Iris Dataset")
    histogramas_distribucion(df_iris, caracteristicas_iris, "Iris Dataset")
    pairplot_iris(df_iris)
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("GUARDANDO RESULTADOS...")
    print("="*80)
    
    # Guardar estadísticas en CSV
    stats_globales_hojas.to_csv('estadisticas_globales_hojas.csv', index=False)
    stats_clase_hojas.to_csv('estadisticas_por_clase_hojas.csv', index=False)
    stats_globales_iris.to_csv('estadisticas_globales_iris.csv', index=False)
    stats_clase_iris.to_csv('estadisticas_por_clase_iris.csv', index=False)
    
    print("\n✓ Estadísticas guardadas en archivos CSV")
    print("✓ Visualizaciones guardadas en archivos PNG")
    print("\n" + "="*80)
    print("ANÁLISIS EXPLORATORIO COMPLETADO")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
