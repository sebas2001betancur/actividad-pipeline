import pandas as pd
import sys
import os

def test_archivo_existe():
    assert os.path.exists("sdss_sample.csv"), "ERROR: sdss_sample.csv no encontrado"
    print("OK - Archivo sdss_sample.csv encontrado")

def test_columnas():
    df = pd.read_csv("sdss_sample.csv")
    columnas_requeridas = ['u', 'g', 'r', 'i', 'z', 'redshift', 'class']
    for col in columnas_requeridas:
        assert col in df.columns, f"ERROR: columna '{col}' no encontrada"
    print(f"OK - Todas las columnas requeridas presentes: {columnas_requeridas}")

def test_nulos():
    df = pd.read_csv("sdss_sample.csv")
    nulos = df.isnull().sum().sum()
    assert nulos == 0, f"ERROR: hay {nulos} valores nulos en el dataset"
    print(f"OK - Sin valores nulos ({nulos})")

def test_clases():
    df = pd.read_csv("sdss_sample.csv")
    clases = set(df['class'].unique())
    clases_esperadas = {'Galaxy', 'Star', 'QSO'}
    assert clases == clases_esperadas, f"ERROR: clases inesperadas {clases}"
    print(f"OK - Clases correctas: {clases}")

def test_tamano():
    df = pd.read_csv("sdss_sample.csv")
    assert len(df) >= 100, f"ERROR: dataset muy pequeño ({len(df)} filas)"
    print(f"OK - Tamano del dataset: {len(df)} filas")

def test_tipos_numericos():
    df = pd.read_csv("sdss_sample.csv")
    cols_numericas = ['u', 'g', 'r', 'i', 'z', 'redshift']
    for col in cols_numericas:
        assert pd.api.types.is_numeric_dtype(df[col]), f"ERROR: {col} no es numerica"
    print(f"OK - Todas las columnas numericas tienen tipo correcto")

if __name__ == "__main__":
    print("Ejecutando pruebas del dataset...\n")
    errores = 0
    pruebas = [
        test_archivo_existe,
        test_columnas,
        test_nulos,
        test_clases,
        test_tamano,
        test_tipos_numericos
    ]
    for prueba in pruebas:
        try:
            prueba()
        except AssertionError as e:
            print(e)
            errores += 1
        except Exception as e:
            print(f"ERROR inesperado en {prueba.__name__}: {e}")
            errores += 1

    print(f"\n{len(pruebas) - errores}/{len(pruebas)} pruebas pasaron")
    if errores > 0:
        sys.exit(1)
    else:
        print("Todas las pruebas pasaron correctamente")
