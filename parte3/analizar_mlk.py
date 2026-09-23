"""
analizar_mlk.py
Parte 3 - Análisis retórico y de N-gramas en el discurso de Martin Luther King Jr.
"""

import re
from collections import Counter
from pathlib import Path
import pandas as pd

# Busca el archivo txt en la misma carpeta donde esté este script
CORPUS_PATH = Path(__file__).parent / "discurso_mlk.txt"

def cargar_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del corpus en: {path.resolve()}")
    
    contenido = path.read_text(encoding="utf-8")
    
    # Extraer el texto limpio quitando las etiquetas de cabecera
    patron = re.compile(r"===\s*TEMA:.*?===\s*(.*?)===\s*FIN\s*===", re.DOTALL | re.IGNORECASE)
    match = patron.search(contenido)
    
    if match:
        return match.group(1).strip()
    return contenido.strip()

def tokenizar_con_puntuacion(texto):
    # Convertimos a minúsculas y limpiamos caracteres extraños manteniendo palabras completas
    # NOTA: No quitamos stopwords porque en retórica los conectores y pronombres forman las frases repetidas.
    return re.findall(r"[a-z']+", texto.lower())

def construir_ngramas(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def generar_tabla_ngramas(tokens, n, top=15):
    grams = construir_ngramas(tokens, n)
    total = len(grams)
    counts = Counter(grams)

    filas = [{
        "ngrama": " ".join(g),
        "frecuencia": c,
        "frecuencia_relativa": round(c / total, 5) if total else 0
    } for g, c in counts.most_common(top)]

    return pd.DataFrame(filas)

def main():
    try:
        texto = cargar_corpus(CORPUS_PATH)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    tokens = tokenizar_con_puntuacion(texto)
    print(tokens)
    
    print("=" * 60)
    print("ANALISIS RETORICO: DISCURSO MARTIN LUTHER KING JR.")
    print("=" * 60)
    print(f"Total de palabras (tokens): {len(tokens)}")
    print(f"Vocabulario único: {len(set(tokens))}")

    # 1. Calcular Bigramas
    print("\n" + "-" * 50)
    print("TOP 15 BIGRAMAS MAS FRECUENTES (Estructura de estilo)")
    print("-" * 50)
    df_bigramas = generar_tabla_ngramas(tokens, 2, 15)
    print(df_bigramas.to_string(index=False))

    # 2. Calcular Trigramas
    print("\n" + "-" * 50)
    print("TOP 15 TRIGRAMAS MAS FRECUENTES (Anáforas y Repeticiones)")
    print("-" * 50)
    df_trigramas = generar_tabla_ngramas(tokens, 3, 15)
    print(df_trigramas.to_string(index=False))

    # 3. Guardar resultados en archivos CSV
    out_dir = Path(__file__).parent / "resultados_mlk"
    out_dir.mkdir(exist_ok=True)
    
    df_bigramas.to_csv(out_dir / "mlk_bigramas.csv", index=False, encoding="utf-8-sig")
    df_trigramas.to_csv(out_dir / "mlk_trigramas.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Análisis completado. Archivos CSV guardados en: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
