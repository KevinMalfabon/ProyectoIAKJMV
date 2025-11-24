"""
Proyecto Final - Razonamiento con IA
Parte 2: Algoritmos de Clustering (Aprendizaje No Supervisado)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
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
    """Carga el dataset Iris"""
    data = load_iris()
    X = data.data
    y = data.target
    nombres_especies = ['Setosa', 'Versicolor', 'Virginica']
    return X, y, nombres_especies

def calcular_pureza(y_true, y_pred):
    """Calcula la pureza de los clusters"""
    n_clusters = len(np.unique(y_pred))
    n_samples = len(y_true)
    pureza_total = 0
    
    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        if np.sum(mask) > 0:
            clases_en_cluster = y_true[mask]
            clase_mayoritaria = np.bincount(clases_en_cluster).max()
            pureza_total += clase_mayoritaria
    
    return pureza_total / n_samples

# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def kmeans_clustering(X, y_true, nombres_especies, nombre_dataset, n_clusters=3):
    """Aplica K-Means clustering y evalúa resultados"""
    
    print(f"\n{'='*80}")
    print(f"K-MEANS CLUSTERING - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probar diferentes valores de k
    print("Explorando diferentes valores de k...\n")
    silhouette_scores = {}
    inertias = {}
    
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50, 
                       max_iter=300, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = silhouette
        inertias[k] = kmeans.inertia_
        print(f"k={k}: Silhouette Score = {silhouette:.4f}, Inertia = {kmeans.inertia_:.2f}")
    
    # Aplicar K-Means con k óptimo
    print(f"\n--- Aplicando K-Means con k={n_clusters} ---\n")
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, 
                   max_iter=300, random_state=42)
    y_pred = kmeans.fit_predict(X_scaled)
    
    # Métricas
    silhouette = silhouette_score(X_scaled, y_pred)
    pureza = calcular_pureza(y_true, y_pred)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Iteraciones hasta convergencia: {kmeans.n_iter_}")
    print(f"Pureza: {pureza*100:.2f}%")
    
    # Análisis de pureza por cluster
    print("\n--- Análisis de Clusters ---\n")
    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        clases_en_cluster = y_true[mask]
        print(f"Cluster {cluster_id}:")
        print(f"  Tamaño: {np.sum(mask)} instancias")
        for clase_id, nombre_clase in enumerate(nombres_especies):
            count = np.sum(clases_en_cluster == clase_id)
            porcentaje = (count / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0
            print(f"  {nombre_clase}: {count} ({porcentaje:.1f}%)")
    
    # Visualizaciones
    visualizar_kmeans(X, y_true, y_pred, kmeans, scaler, nombres_especies, 
                     nombre_dataset, silhouette_scores)
    
    return {
        'model': kmeans,
        'labels': y_pred,
        'silhouette': silhouette,
        'pureza': pureza,
        'inertia': kmeans.inertia_,
        'silhouette_scores': silhouette_scores
    }

def visualizar_kmeans(X, y_true, y_pred, kmeans, scaler, nombres_especies, 
                     nombre_dataset, silhouette_scores):
    """Visualiza resultados de K-Means"""
    
    # Figura con múltiples subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Clusters encontrados
    ax1 = plt.subplot(2, 3, 1)
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for cluster_id in range(len(np.unique(y_pred))):
        mask = y_pred == cluster_id
        ax1.scatter(X[mask, 0], X[mask, 1], c=colores[cluster_id], 
                   s=100, alpha=0.6, edgecolors='black', linewidth=1,
                   label=f'Cluster {cluster_id}')
    
    # Centroides
    centroides = scaler.inverse_transform(kmeans.cluster_centers_)
    ax1.scatter(centroides[:, 0], centroides[:, 1], c='red', 
               marker='X', s=300, edgecolors='black', linewidth=2,
               label='Centroides')
    
    ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax1.set_title('Clusters Encontrados', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Clases reales
    ax2 = plt.subplot(2, 3, 2)
    for clase_id, nombre_clase in enumerate(nombres_especies):
        mask = y_true == clase_id
        ax2.scatter(X[mask, 0], X[mask, 1], c=colores[clase_id], 
                   s=100, alpha=0.6, edgecolors='black', linewidth=1,
                   label=nombre_clase)
    
    ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax2.set_title('Clases Reales', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Silhouette Score vs k
    ax3 = plt.subplot(2, 3, 3)
    k_values = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    ax3.plot(k_values, scores, marker='o', linewidth=2, markersize=10, color='#FF6B6B')
    ax3.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax3.set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(k_values)
    
    # 4. Silhouette Plot
    ax4 = plt.subplot(2, 3, 4)
    X_scaled = scaler.transform(X)
    silhouette_vals = silhouette_samples(X_scaled, y_pred)
    silhouette_avg = silhouette_score(X_scaled, y_pred)
    
    y_lower = 10
    for cluster_id in range(len(np.unique(y_pred))):
        cluster_silhouette_vals = silhouette_vals[y_pred == cluster_id]
        cluster_silhouette_vals.sort()
        
        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster
        
        ax4.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=colores[cluster_id], edgecolor=colores[cluster_id], 
                         alpha=0.7)
        ax4.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
        y_lower = y_upper + 10
    
    ax4.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
               label=f'Promedio: {silhouette_avg:.3f}')
    ax4.set_xlabel('Coeficiente de Silueta', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax4.set_title('Silhouette Plot', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # 5. Comparación Clusters vs Clases
    ax5 = plt.subplot(2, 3, 5)
    marcadores = ['o', 's', '^']
    for clase_id, nombre_clase in enumerate(nombres_especies):
        mask = y_true == clase_id
        ax5.scatter(X[mask, 0], X[mask, 1], c=[colores[c] for c in y_pred[mask]], 
                   marker=marcadores[clase_id], s=100, alpha=0.6, 
                   edgecolors='black', linewidth=1, label=nombre_clase)
    
    ax5.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax5.set_title('Clusters (color) vs Clases (forma)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Matriz de confusión (Clusters vs Clases)
    ax6 = plt.subplot(2, 3, 6)
    confusion = np.zeros((len(nombres_especies), len(np.unique(y_pred))))
    for i in range(len(nombres_especies)):
        for j in range(len(np.unique(y_pred))):
            confusion[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Blues', ax=ax6,
               xticklabels=[f'Cluster {i}' for i in range(len(np.unique(y_pred)))],
               yticklabels=nombres_especies, cbar_kws={'label': 'Count'})
    ax6.set_title('Matriz de Asignación\n(Clases Reales vs Clusters)', 
                 fontsize=14, fontweight='bold')
    ax6.set_ylabel('Clase Real', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Cluster Asignado', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'K-Means Clustering - {nombre_dataset}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'kmeans_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# CLUSTERING JERÁRQUICO
# ============================================================================

def clustering_jerarquico(X, y_true, nombres_especies, nombre_dataset, n_clusters=3):
    """Aplica clustering jerárquico y evalúa resultados"""
    
    print(f"\n{'='*80}")
    print(f"CLUSTERING JERÁRQUICO - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Probar diferentes métodos de enlace
    print("Explorando diferentes métodos de enlace...\n")
    metodos = ['ward', 'complete', 'average', 'single']
    resultados_metodos = {}
    
    for metodo in metodos:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=metodo)
        labels = agg.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        resultados_metodos[metodo] = {
            'silhouette': silhouette,
            'labels': labels
        }
        print(f"{metodo.capitalize():12s}: Silhouette Score = {silhouette:.4f}")
    
    # Usar el mejor método
    mejor_metodo = max(resultados_metodos, key=lambda x: resultados_metodos[x]['silhouette'])
    print(f"\nMejor método: {mejor_metodo}")
    
    # Aplicar clustering jerárquico con el mejor método
    print(f"\n--- Aplicando Clustering Jerárquico (método: {mejor_metodo}) ---\n")
    
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=mejor_metodo)
    y_pred = agg.fit_predict(X_scaled)
    
    # Métricas
    silhouette = silhouette_score(X_scaled, y_pred)
    pureza = calcular_pureza(y_true, y_pred)
    
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Pureza: {pureza*100:.2f}%")
    
    # Análisis de pureza por cluster
    print("\n--- Análisis de Clusters ---\n")
    for cluster_id in range(n_clusters):
        mask = y_pred == cluster_id
        clases_en_cluster = y_true[mask]
        print(f"Cluster {cluster_id}:")
        print(f"  Tamaño: {np.sum(mask)} instancias")
        for clase_id, nombre_clase in enumerate(nombres_especies):
            count = np.sum(clases_en_cluster == clase_id)
            porcentaje = (count / np.sum(mask)) * 100 if np.sum(mask) > 0 else 0
            print(f"  {nombre_clase}: {count} ({porcentaje:.1f}%)")
    
    # Visualizaciones
    visualizar_jerarquico(X, X_scaled, y_true, y_pred, nombres_especies, 
                         nombre_dataset, mejor_metodo, resultados_metodos)
    
    return {
        'model': agg,
        'labels': y_pred,
        'silhouette': silhouette,
        'pureza': pureza,
        'mejor_metodo': mejor_metodo,
        'resultados_metodos': resultados_metodos
    }

def visualizar_jerarquico(X, X_scaled, y_true, y_pred, nombres_especies, 
                         nombre_dataset, metodo, resultados_metodos):
    """Visualiza resultados de clustering jerárquico"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Dendrograma
    ax1 = plt.subplot(2, 3, 1)
    linkage_matrix = linkage(X_scaled, method=metodo)
    dendrogram(linkage_matrix, ax=ax1, color_threshold=0, above_threshold_color='black')
    ax1.set_title('Dendrograma', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Índice de Muestra', fontsize=12)
    ax1.set_ylabel('Distancia', fontsize=12)
    ax1.axhline(y=linkage_matrix[-2, 2], color='red', linestyle='--', linewidth=2,
               label='Corte para 3 clusters')
    ax1.legend()
    
    # 2. Clusters encontrados
    ax2 = plt.subplot(2, 3, 2)
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for cluster_id in range(len(np.unique(y_pred))):
        mask = y_pred == cluster_id
        ax2.scatter(X[mask, 0], X[mask, 1], c=colores[cluster_id], 
                   s=100, alpha=0.6, edgecolors='black', linewidth=1,
                   label=f'Cluster {cluster_id}')
    
    ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax2.set_title('Clusters Encontrados', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Clases reales
    ax3 = plt.subplot(2, 3, 3)
    for clase_id, nombre_clase in enumerate(nombres_especies):
        mask = y_true == clase_id
        ax3.scatter(X[mask, 0], X[mask, 1], c=colores[clase_id], 
                   s=100, alpha=0.6, edgecolors='black', linewidth=1,
                   label=nombre_clase)
    
    ax3.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax3.set_title('Clases Reales', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparación de métodos de enlace
    ax4 = plt.subplot(2, 3, 4)
    metodos = list(resultados_metodos.keys())
    scores = [resultados_metodos[m]['silhouette'] for m in metodos]
    bars = ax4.bar(metodos, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax4.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax4.set_title('Comparación de Métodos de Enlace', fontsize=14, fontweight='bold')
    ax4.set_ylim([min(scores) - 0.05, max(scores) + 0.05])
    
    # Añadir valores sobre las barras
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Silhouette Plot
    ax5 = plt.subplot(2, 3, 5)
    silhouette_vals = silhouette_samples(X_scaled, y_pred)
    silhouette_avg = silhouette_score(X_scaled, y_pred)
    
    y_lower = 10
    for cluster_id in range(len(np.unique(y_pred))):
        cluster_silhouette_vals = silhouette_vals[y_pred == cluster_id]
        cluster_silhouette_vals.sort()
        
        size_cluster = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster
        
        ax5.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                         facecolor=colores[cluster_id], edgecolor=colores[cluster_id], 
                         alpha=0.7)
        ax5.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster_id))
        y_lower = y_upper + 10
    
    ax5.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
               label=f'Promedio: {silhouette_avg:.3f}')
    ax5.set_xlabel('Coeficiente de Silueta', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax5.set_title('Silhouette Plot', fontsize=14, fontweight='bold')
    ax5.legend()
    
    # 6. Matriz de confusión
    ax6 = plt.subplot(2, 3, 6)
    confusion = np.zeros((len(nombres_especies), len(np.unique(y_pred))))
    for i in range(len(nombres_especies)):
        for j in range(len(np.unique(y_pred))):
            confusion[i, j] = np.sum((y_true == i) & (y_pred == j))
    
    sns.heatmap(confusion, annot=True, fmt='.0f', cmap='Greens', ax=ax6,
               xticklabels=[f'Cluster {i}' for i in range(len(np.unique(y_pred)))],
               yticklabels=nombres_especies, cbar_kws={'label': 'Count'})
    ax6.set_title('Matriz de Asignación\n(Clases Reales vs Clusters)', 
                 fontsize=14, fontweight='bold')
    ax6.set_ylabel('Clase Real', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Cluster Asignado', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Clustering Jerárquico - {nombre_dataset}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'jerarquico_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# COMPARACIÓN DE ALGORITMOS DE CLUSTERING
# ============================================================================

def comparar_clustering(resultados_kmeans, resultados_jerarquico, nombre_dataset):
    """Compara los resultados de K-Means y Clustering Jerárquico"""
    
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN DE ALGORITMOS DE CLUSTERING - {nombre_dataset}")
    print(f"{'='*80}\n")
    
    # Tabla comparativa
    comparacion = pd.DataFrame({
        'Algoritmo': ['K-Means', 'Clustering Jerárquico'],
        'Silhouette Score': [
            resultados_kmeans['silhouette'],
            resultados_jerarquico['silhouette']
        ],
        'Pureza (%)': [
            resultados_kmeans['pureza'] * 100,
            resultados_jerarquico['pureza'] * 100
        ]
    })
    
    print(comparacion.to_string(index=False))
    
    # Visualización comparativa
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Silhouette Score
    algoritmos = ['K-Means', 'Clustering\nJerárquico']
    scores = [resultados_kmeans['silhouette'], resultados_jerarquico['silhouette']]
    bars1 = axes[0].bar(algoritmos, scores, color=['#FF6B6B', '#4ECDC4'], 
                       edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Comparación de Silhouette Score', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(scores) - 0.05, max(scores) + 0.05])
    
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Pureza
    purezas = [resultados_kmeans['pureza'] * 100, resultados_jerarquico['pureza'] * 100]
    bars2 = axes[1].bar(algoritmos, purezas, color=['#FF6B6B', '#4ECDC4'], 
                       edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Pureza (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Comparación de Pureza', fontsize=14, fontweight='bold')
    axes[1].set_ylim([min(purezas) - 5, 100])
    
    for bar, pureza in zip(bars2, purezas):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{pureza:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle(f'Comparación de Algoritmos de Clustering - {nombre_dataset}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'comparacion_clustering_{nombre_dataset.lower().replace(" ", "_")}.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparacion

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el análisis de clustering"""
    
    print("\n" + "="*80)
    print("PROYECTO FINAL - RAZONAMIENTO CON IA")
    print("PARTE 2: ALGORITMOS DE CLUSTERING (APRENDIZAJE NO SUPERVISADO)")
    print("="*80)
    
    # ========================================================================
    # CLUSTERING EN LA BASE DE HOJAS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# CLUSTERING EN LA BASE DE DATOS DE HOJAS")
    print("#"*80)
    
    X_hojas, y_hojas, nombres_hojas = cargar_datos_hojas()
    
    # K-Means
    resultados_kmeans_hojas = kmeans_clustering(X_hojas, y_hojas, nombres_hojas, 
                                                "Base de Hojas", n_clusters=3)
    
    # Clustering Jerárquico
    resultados_jerarquico_hojas = clustering_jerarquico(X_hojas, y_hojas, nombres_hojas, 
                                                        "Base de Hojas", n_clusters=3)
    
    # Comparación
    comparacion_hojas = comparar_clustering(resultados_kmeans_hojas, 
                                           resultados_jerarquico_hojas, 
                                           "Base de Hojas")
    
    # ========================================================================
    # CLUSTERING EN LA BASE DE IRIS
    # ========================================================================
    
    print("\n\n" + "#"*80)
    print("# CLUSTERING EN LA BASE DE DATOS IRIS")
    print("#"*80)
    
    X_iris, y_iris, nombres_iris = cargar_datos_iris()
    
    # K-Means
    resultados_kmeans_iris = kmeans_clustering(X_iris, y_iris, nombres_iris, 
                                              "Iris Dataset", n_clusters=3)
    
    # Clustering Jerárquico
    resultados_jerarquico_iris = clustering_jerarquico(X_iris, y_iris, nombres_iris, 
                                                       "Iris Dataset", n_clusters=3)
    
    # Comparación
    comparacion_iris = comparar_clustering(resultados_kmeans_iris, 
                                          resultados_jerarquico_iris, 
                                          "Iris Dataset")
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    
    print("\n\n" + "="*80)
    print("GUARDANDO RESULTADOS...")
    print("="*80)
    
    comparacion_hojas.to_csv('comparacion_clustering_hojas.csv', index=False)
    comparacion_iris.to_csv('comparacion_clustering_iris.csv', index=False)
    
    print("\n✓ Resultados guardados en archivos CSV")
    print("✓ Visualizaciones guardadas en archivos PNG")
    print("\n" + "="*80)
    print("ANÁLISIS DE CLUSTERING COMPLETADO")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
