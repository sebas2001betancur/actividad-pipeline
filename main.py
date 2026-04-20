import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # para que funcione sin pantalla en Docker
import matplotlib.pyplot as plt
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, mean_squared_error, r2_score)

# crear carpeta de salida si no existe
os.makedirs("outputs", exist_ok=True)


def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    print(f"Dataset cargado: {df.shape}")
    print(f"Clases: {df['class'].value_counts().to_dict()}")
    print(f"Nulos: {df.isnull().sum().sum()}")
    return df

def preparar_datos(df):
    features = ['u', 'g', 'r', 'i', 'z', 'redshift']
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    print(f"Clases codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X, y, le


#  CLASIFICACION KNN

def clasificacion_knn(df):
    print("\n" + "="*50)
    print("1. CLASIFICACION KNN")
    print("="*50)

    features = ['u', 'g', 'r', 'i', 'z', 'redshift']
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['class'])

    # division 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    # modelo kNN con k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Matriz de confusion:\n{cm}")

    # grafica
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Clasificacion KNN (k=5)\nAccuracy: {acc:.4f}')
    plt.tight_layout()
    plt.savefig('outputs/clasificacion_confusion_matrix.png', dpi=120)
    plt.close()

    # guardar metrica
    metricas = {"modelo": "KNN", "k": 5, "accuracy": round(acc, 4),
                "train_size": len(X_train), "test_size": len(X_test)}
    return metricas


# REGRESION LINEAL

def regresion_lineal(df):
    print("\n" + "="*50)
    print("2. REGRESION LINEAL")
    print("="*50)

    # variables de entrada: u g r i z
    # variable objetivo: redshift
    X = df[['u', 'g', 'r', 'i', 'z']]
    y = df['redshift']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_train_sc, y_train)
    y_pred = modelo.predict(X_test_sc)

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    # grafica real vs predicho
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Regresion Lineal - Prediccion de Redshift', fontweight='bold')

    axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lim, lim, 'r--', linewidth=1.5, label='Prediccion perfecta')
    axes[0].set_title(f'Real vs Predicho\nR2={r2:.4f} | RMSE={rmse:.4f}')
    axes[0].set_xlabel('Redshift real')
    axes[0].set_ylabel('Redshift predicho')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuos = y_test.values - y_pred
    axes[1].hist(residuos, bins=40, color='steelblue', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title('Distribucion de residuos')
    axes[1].set_xlabel('Residuo')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/regresion_resultados.png', dpi=120)
    plt.close()

    metricas = {"modelo": "RegresionLineal", "mse": round(mse, 4),
                "rmse": round(rmse, 4), "r2": round(r2, 4)}
    return metricas


# CLUSTERING KMEANS

def clustering_kmeans(df):
    print("\n" + "="*50)
    print("3. CLUSTERING KMEANS")
    print("="*50)

    magnitudes = ['u', 'g', 'r', 'i', 'z']
    X = df[magnitudes]

    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_sc)
    df['cluster'] = clusters

    # clases reales para comparar
    le = LabelEncoder()
    clases_reales = le.fit_transform(df['class'])

    print(f"Distribucion de clusters: {pd.Series(clusters).value_counts().to_dict()}")

    # grafica: clusters vs clases reales
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('KMeans Clustering - Magnitudes Fotometricas', fontweight='bold')

    colores_cluster = ['steelblue', 'orange', 'green']
    colores_clase   = ['red', 'blue', 'purple']
##docker build -t ml-pipeline-sdss .
##docker run -v $(pwd)/outputs:/app/outputs ml-pipeline-sdss

    # clusters obtenidos
    for c in range(3):
        mask = clusters == c
        axes[0].scatter(X_sc[mask, 0], X_sc[mask, 2],
                        alpha=0.4, s=15, color=colores_cluster[c],
                        label=f'Cluster {c}')
    axes[0].set_title('Clusters obtenidos por KMeans')
    axes[0].set_xlabel('u (normalizada)')
    axes[0].set_ylabel('r (normalizada)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # clases reales
    for c, nombre in enumerate(le.classes_):
        mask = clases_reales == c
        axes[1].scatter(X_sc[mask, 0], X_sc[mask, 2],
                        alpha=0.4, s=15, color=colores_clase[c],
                        label=nombre)
    axes[1].set_title('Clases reales (Galaxy, QSO, Star)')
    axes[1].set_xlabel('u (normalizada)')
    axes[1].set_ylabel('r (normalizada)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/clustering_kmeans.png', dpi=120)
    plt.close()

    metricas = {"modelo": "KMeans", "n_clusters": 3,
                "distribucion": pd.Series(clusters).value_counts().to_dict()}
    return metricas


# GUARDAR METRICAS

def guardar_metricas(m_cls, m_reg, m_clu):
    metricas = {
        "clasificacion": m_cls,
        "regresion": m_reg,
        "clustering": m_clu
    }
    with open("outputs/metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("\nMetricas guardadas en outputs/metricas.json")

    # resumen en txt
    with open("outputs/resumen.txt", "w") as f:
        f.write("RESUMEN PIPELINE ML - SDSS\n")
        f.write("="*40 + "\n\n")
        f.write(f"CLASIFICACION KNN (k=5)\n")
        f.write(f"  Accuracy: {m_cls['accuracy']}\n\n")
        f.write(f"REGRESION LINEAL\n")
        f.write(f"  MSE:  {m_reg['mse']}\n")
        f.write(f"  RMSE: {m_reg['rmse']}\n")
        f.write(f"  R2:   {m_reg['r2']}\n\n")
        f.write(f"CLUSTERING KMEANS (k=3)\n")
        f.write(f"  Distribucion: {m_clu['distribucion']}\n")
    print("Resumen guardado en outputs/resumen.txt")

# MAIN

if __name__ == "__main__":
    print("Iniciando pipeline de ML...")
    df = cargar_datos("sdss_sample.csv")
    m_cls = clasificacion_knn(df)
    m_reg = regresion_lineal(df)
    m_clu = clustering_kmeans(df)
    guardar_metricas(m_cls, m_reg, m_clu)
    print("\nPipeline completado. Resultados en outputs/")
