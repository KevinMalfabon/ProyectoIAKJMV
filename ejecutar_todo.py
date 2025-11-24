"""
Proyecto Final - Razonamiento con IA
Script Maestro: Ejecuta todo el proyecto de principio a fin
"""

import sys
import time

def imprimir_seccion(titulo):
    """Imprime un título de sección formateado"""
    print("\n\n" + "="*80)
    print(f"  {titulo}")
    print("="*80 + "\n")
    time.sleep(1)

def main():
    """Ejecuta todos los scripts en orden"""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "PROYECTO FINAL - RAZONAMIENTO CON IA" + " "*22 + "█")
    print("█" + " "*15 + "Ejecución Completa de Todos los Análisis" + " "*24 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    inicio_total = time.time()
    
    # ========================================================================
    # PARTE 1: EXPLORACIÓN DE DATOS
    # ========================================================================
    
    imprimir_seccion("PARTE 1/5: EXPLORACIÓN Y ANÁLISIS DESCRIPTIVO DE DATOS")
    
    try:
        inicio = time.time()
        import importlib
        modulo1 = importlib.import_module('1_exploracion_datos')
        modulo1.main()
        tiempo1 = time.time() - inicio
        print(f"\n✓ Parte 1 completada en {tiempo1:.2f} segundos")
    except Exception as e:
        print(f"\n✗ Error en Parte 1: {e}")
        return
    
    # ========================================================================
    # PARTE 2: CLUSTERING
    # ========================================================================
    
    imprimir_seccion("PARTE 2/5: ALGORITMOS DE CLUSTERING (NO SUPERVISADO)")
    
    try:
        inicio = time.time()
        modulo2 = importlib.import_module('2_clustering')
        modulo2.main()
        tiempo2 = time.time() - inicio
        print(f"\n✓ Parte 2 completada en {tiempo2:.2f} segundos")
    except Exception as e:
        print(f"\n✗ Error en Parte 2: {e}")
        return
    
    # ========================================================================
    # PARTE 3: CLASIFICACIÓN SUPERVISADA
    # ========================================================================
    
    imprimir_seccion("PARTE 3/5: ALGORITMOS DE CLASIFICACIÓN SUPERVISADA")
    print("NOTA: Esta parte puede tomar varios minutos debido a GridSearchCV...\n")
    
    try:
        inicio = time.time()
        modulo3 = importlib.import_module('3_clasificacion_supervisada')
        modulo3.main()
        tiempo3 = time.time() - inicio
        print(f"\n✓ Parte 3 completada en {tiempo3:.2f} segundos")
    except Exception as e:
        print(f"\n✗ Error en Parte 3: {e}")
        return
    
    # ========================================================================
    # PARTE 4: VISUALIZACIONES
    # ========================================================================
    
    imprimir_seccion("PARTE 4/5: VISUALIZACIONES DE FRONTERAS Y MÉTRICAS")
    
    try:
        inicio = time.time()
        modulo4 = importlib.import_module('4_visualizaciones')
        modulo4.main()
        tiempo4 = time.time() - inicio
        print(f"\n✓ Parte 4 completada en {tiempo4:.2f} segundos")
    except Exception as e:
        print(f"\n✗ Error en Parte 4: {e}")
        return
    
    # ========================================================================
    # PARTE 5: ANÁLISIS COMPARATIVO
    # ========================================================================
    
    imprimir_seccion("PARTE 5/5: ANÁLISIS COMPARATIVO FINAL")
    
    try:
        inicio = time.time()
        modulo5 = importlib.import_module('5_analisis_comparativo')
        modulo5.main()
        tiempo5 = time.time() - inicio
        print(f"\n✓ Parte 5 completada en {tiempo5:.2f} segundos")
    except Exception as e:
        print(f"\n✗ Error en Parte 5: {e}")
        return
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    tiempo_total = time.time() - inicio_total
    
    print("\n\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*25 + "PROYECTO COMPLETADO" + " "*34 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    print("RESUMEN DE TIEMPOS:")
    print(f"  Parte 1 (Exploración):        {tiempo1:8.2f} segundos")
    print(f"  Parte 2 (Clustering):         {tiempo2:8.2f} segundos")
    print(f"  Parte 3 (Clasificación):      {tiempo3:8.2f} segundos")
    print(f"  Parte 4 (Visualizaciones):    {tiempo4:8.2f} segundos")
    print(f"  Parte 5 (Análisis):           {tiempo5:8.2f} segundos")
    print(f"  " + "-"*50)
    print(f"  TIEMPO TOTAL:                 {tiempo_total:8.2f} segundos ({tiempo_total/60:.2f} minutos)")
    
    print("\n" + "="*80)
    print("ARCHIVOS GENERADOS:")
    print("="*80)
    print("\n📊 TABLAS CSV:")
    print("  - estadisticas_globales_hojas.csv")
    print("  - estadisticas_por_clase_hojas.csv")
    print("  - estadisticas_globales_iris.csv")
    print("  - estadisticas_por_clase_iris.csv")
    print("  - comparacion_clustering_hojas.csv")
    print("  - comparacion_clustering_iris.csv")
    print("  - ranking_clasificacion_hojas.csv")
    print("  - ranking_clasificacion_iris.csv")
    print("  - tabla_completa_hojas.csv")
    print("  - tabla_completa_iris.csv")
    print("  - tabla_comparacion_datasets.csv")
    print("  - tabla_hiperparametros_hojas.csv")
    print("  - tabla_hiperparametros_iris.csv")
    
    print("\n🖼️  VISUALIZACIONES PNG:")
    print("  - Más de 30 gráficos de alta calidad generados")
    print("  - Fronteras de decisión, matrices de confusión, comparaciones, etc.")
    
    print("\n📦 MODELOS:")
    print("  - modelos_hojas.pkl")
    print("  - modelos_iris.pkl")
    
    print("\n📄 REPORTES:")
    print("  - REPORTE_FINAL.txt")
    
    print("\n" + "="*80)
    print("¡PROYECTO FINALIZADO EXITOSAMENTE!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
