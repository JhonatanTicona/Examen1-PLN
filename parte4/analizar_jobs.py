"""
analizar_jobs.py
Parte 4 - Análisis lingüístico y de N-gramas en el discurso de Steve Jobs en Stanford.
"""

import re
from collections import Counter
from pathlib import Path
import pandas as pd

# Busca el archivo txt en la misma carpeta donde esté este script
CORPUS_PATH = Path(__file__).parent / "discurso_jobs.txt"

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

def tokenizar(texto):
    # Convertimos a minúsculas y extraemos palabras puras (incluyendo contracciones como don't)
    return re.findall(r"[a-z']+", texto.lower())

def construir_ngramas(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def generar_tabla_ngramas(tokens, n, top=20):
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

    tokens = tokenizar(texto)
    print(tokens)
    print("=" * 60)
    print("ANALISIS LINGÜISTICO: STEVE JOBS - STANFORD")
    print("=" * 60)
    print(f"Total de palabras (tokens): {len(tokens)}")
    print(f"Vocabulario único: {len(set(tokens))}")

    # 1. Calcular 20 Unigramas
    print("\n" + "-" * 50)
    print("TOP 20 UNIGRAMAS MAS FRECUENTES")
    print("-" * 50)
    df_unigramas = generar_tabla_ngramas(tokens, 1, 20)
    print(df_unigramas.to_string(index=False))

    # 2. Calcular 20 Bigramas
    print("\n" + "-" * 50)
    print("TOP 20 BIGRAMAS MAS FRECUENTES")
    print("-" * 50)
    df_bigramas = generar_tabla_ngramas(tokens, 2, 20)
    print(df_bigramas.to_string(index=False))

    # 3. Calcular 20 Trigramas
    print("\n" + "-" * 50)
    print("TOP 20 TRIGRAMAS MAS FRECUENTES")
    print("-" * 50)
    df_trigramas = generar_tabla_ngramas(tokens, 3, 20)
    print(df_trigramas.to_string(index=False))

    # 4. Guardar resultados en la carpeta resultados_jobs
    out_dir = Path(__file__).parent / "resultados_jobs"
    out_dir.mkdir(exist_ok=True)
    
    df_unigramas.to_csv(out_dir / "jobs_unigramas.csv", index=False, encoding="utf-8-sig")
    df_bigramas.to_csv(out_dir / "jobs_bigramas.csv", index=False, encoding="utf-8-sig")
    df_trigramas.to_csv(out_dir / "jobs_trigramas.csv", index=False, encoding="utf-8-sig")
    
    print(f"\n✅ Análisis completado. Archivos CSV guardados en: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
