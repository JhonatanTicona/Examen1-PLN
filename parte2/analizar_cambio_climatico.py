"""
analizar_cambio_climatico.py
Parte 2 - N-gramas sobre cambio climático.

Compara:
A = periodo anterior (2011-2014)
B = periodo reciente (2023-2026)

Calcula vocabulario, unigramas, bigramas, trigramas,
frecuencia relativa y términos/expresiones solicitados.
"""

import re
from collections import Counter
from pathlib import Path
import pandas as pd

CORPUS_PATH = Path(__file__).parent / "corpus_cambio_climatico.txt"

STOPWORDS = {
    "el","la","los","las","un","una","unos","unas","y","o","e","u","de","del",
    "al","a","ante","bajo","con","contra","desde","en","entre","hacia","hasta",
    "para","por","sin","sobre","tras","que","se","su","sus","es","son","ser",
    "ha","han","hay","como","más","muy","también","esta","este","estas","estos",
    "esa","ese","esas","esos","lo","le","les","ya","no","pero","sino","cuando",
    "donde","quienes","quien","cada","puede","pueden","debe","deben","nuestro",
    "nuestra","nuestros","nuestras","algunos","algunas","otro","otra","otros",
    "otras","también","porque","para","entre"
}

TERMINOS = [
    "cambio climático",
    "calentamiento global",
    "emisiones",
    "transición energética",
    "energías renovables",
    "justicia climática",
    "adaptación",
    "mitigación",
    "carbono",
    "futuro"
]

def cargar_corpus(path: Path) -> list:
    contenido = path.read_text(encoding="utf-8")
    documentos = []
    bloques = re.split(r"===\s*FIN\s*===\s*", contenido, flags=re.IGNORECASE)

    patron = re.compile(
        r"===\s*TEMA:\s*(.*?)\s*\|\s*"
        r"FUENTE:\s*(.*?)\s*\|\s*"
        r"TITULO:\s*(.*?)\s*\|\s*"
        r"FECHA:\s*(.*?)\s*\|\s*"
        r"URL:\s*(.*?)\s*===\s*"
        r"(.*)",
        re.IGNORECASE | re.DOTALL
    )

    for bloque in bloques:
        bloque = bloque.strip()
        if not bloque:
            continue

        m = patron.match(bloque)
        if not m:
            print("⚠️ Bloque no reconocido:")
            print(bloque[:200])
            continue

        tema, fuente, titulo, fecha, url, texto = m.groups()

        documentos.append({
            "tema": tema.strip().lower(),
            "fuente": fuente.strip(),
            "titulo": titulo.strip(),
            "fecha": fecha.strip(),
            "url": url.strip(),
            "texto": texto.strip()
        })

    return documentos

def tokenizar(texto):
    palabras = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    return [p for p in palabras if p not in STOPWORDS]

def normalizar(texto):
    return texto.lower().translate(str.maketrans("áéíóúü", "aeiouu"))

def construir_ngramas(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tabla_ngramas(tokens, n, top=20):
    """top=None (o <= 0) devuelve TODOS los n-gramas, sin cortar."""
    grams = construir_ngramas(tokens, n)
    total = len(grams)
    counts = Counter(grams)
    limite = None if (top is None or top <= 0) else top

    filas = [{
        "ngrama": " ".join(g),
        "frecuencia": c,
        "frecuencia_relativa": round(c / total, 5) if total else 0
    } for g, c in counts.most_common(limite)]

    return pd.DataFrame(filas)

def frecuencia_termino(texto, termino):
    texto_norm = normalizar(texto)
    termino_norm = normalizar(termino)
    return len(re.findall(r"\b" + re.escape(termino_norm) + r"\b", texto_norm))

def tabla_terminos(doc_a, doc_b):
    total_a = len(re.findall(r"[a-záéíóúüñ]+", doc_a.lower()))
    total_b = len(re.findall(r"[a-záéíóúüñ]+", doc_b.lower()))

    filas = []

    for termino in TERMINOS:
        fa = frecuencia_termino(doc_a, termino)
        fb = frecuencia_termino(doc_b, termino)

        filas.append({
            "termino": termino,
            "frecuencia_A": fa,
            "frecuencia_B": fb,
            "frecuencia_relativa_A": round(fa / total_a, 6) if total_a else 0,
            "frecuencia_relativa_B": round(fb / total_b, 6) if total_b else 0
        })

    return pd.DataFrame(filas)

def contar_categoria(tokens, palabras):
    palabras_norm = [normalizar(x) for x in palabras]
    return sum(
        freq for gram, freq in Counter(construir_ngramas(tokens, 2)).items()
        if any(normalizar(p) in palabras_norm for p in gram)
    )

def main():
    documentos = cargar_corpus(CORPUS_PATH)

    print(f"Documentos cargados: {len(documentos)}")

    for d in documentos:
        print(f"  - [{d['fecha']}] {d['titulo']}")

    if len(documentos) < 2:
        print("\n❌ Se necesitan los dos documentos: A y B.")
        return

    doc_a = documentos[0]
    doc_b = documentos[1]

    tokens_a = tokenizar(doc_a["texto"])
    tokens_b = tokenizar(doc_b["texto"])

    print(f"\nTokens corpus A: {len(tokens_a)}")
    print(f"Vocabulario único A: {len(set(tokens_a))}")
    print(f"Tokens corpus B: {len(tokens_b)}")
    print(f"Vocabulario único B: {len(set(tokens_b))}")

    # ----------------------------------------------------------
    # 1. Unigramas, bigramas y trigramas
    # ----------------------------------------------------------

    for nombre, tokens in [
        ("CORPUS A - PERIODO ANTERIOR", tokens_a),
        ("CORPUS B - PERIODO RECIENTE", tokens_b)
    ]:
        print("\n" + "=" * 70)
        print(nombre)
        print("=" * 70)

        for n, titulo in [(1, "UNIGRAMAS"), (2, "BIGRAMAS"), (3, "TRIGRAMAS")]:
            print("\n" + "-" * 70)
            print(f"20 {titulo} MAS FRECUENTES")
            print("-" * 70)
            print(tabla_ngramas(tokens, n, 20).to_string(index=False))

    # ----------------------------------------------------------
    # 2. Términos indicados en el parcial
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARACION DE TERMINOS CLAVE")
    print("=" * 70)

    df_terminos = tabla_terminos(doc_a["texto"], doc_b["texto"])
    print(df_terminos.to_string(index=False))

    # ----------------------------------------------------------
    # 3. Acción vs riesgo
    # ----------------------------------------------------------

    palabras_accion = [
        "acción", "reducción", "reducir", "transición", "adaptación",
        "mitigación", "renovables", "descarbonización", "inversión",
        "transformar", "transformación", "políticas"
    ]

    palabras_riesgo = [
        "riesgo", "impactos", "impacto", "pérdidas", "vulnerabilidad",
        "sequías", "inundaciones", "incendios", "olas", "amenazas",
        "daños"
    ]

    accion_a = contar_categoria(tokens_a, palabras_accion)
    accion_b = contar_categoria(tokens_b, palabras_accion)
    riesgo_a = contar_categoria(tokens_a, palabras_riesgo)
    riesgo_b = contar_categoria(tokens_b, palabras_riesgo)

    print("\n" + "=" * 70)
    print("ACCION VS. RIESGO")
    print("=" * 70)
    print(f"Accion - periodo anterior: {accion_a}")
    print(f"Accion - periodo reciente: {accion_b}")
    print(f"Riesgo - periodo anterior: {riesgo_a}")
    print(f"Riesgo - periodo reciente: {riesgo_b}")

    # ----------------------------------------------------------
    # 4. N-gramas relacionados con acción y riesgo
    # ----------------------------------------------------------

    def ngramas_por_categoria(tokens, palabras, top=15):
        palabras_norm = [normalizar(p) for p in palabras]
        grams = construir_ngramas(tokens, 2) + construir_ngramas(tokens, 3)

        filtrados = []
        for g in grams:
            if any(normalizar(p) in palabras_norm for p in g):
                filtrados.append(g)

        return Counter(filtrados).most_common(top)

    print("\n" + "=" * 70)
    print("N-GRAMAS DE ACCION - CORPUS B")
    print("=" * 70)

    for gram, freq in ngramas_por_categoria(tokens_b, palabras_accion):
        print(f"{' '.join(gram):45s} freq={freq}")

    print("\n" + "=" * 70)
    print("N-GRAMAS DE RIESGO - CORPUS B")
    print("=" * 70)

    for gram, freq in ngramas_por_categoria(tokens_b, palabras_riesgo):
        print(f"{' '.join(gram):45s} freq={freq}")

    # ----------------------------------------------------------
    # 5. Guardar CSV
    # ----------------------------------------------------------

    out_dir = Path(__file__).parent / "resultados_clima"
    out_dir.mkdir(exist_ok=True)

    # top=None => TODOS los n-gramas encontrados en cada corpus, sin cortar.
    tabla_ngramas(tokens_a, 1, None).to_csv(
        out_dir / "A_unigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )
    tabla_ngramas(tokens_a, 2, None).to_csv(
        out_dir / "A_bigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )
    tabla_ngramas(tokens_a, 3, None).to_csv(
        out_dir / "A_trigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )

    tabla_ngramas(tokens_b, 1, None).to_csv(
        out_dir / "B_unigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )
    tabla_ngramas(tokens_b, 2, None).to_csv(
        out_dir / "B_bigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )
    tabla_ngramas(tokens_b, 3, None).to_csv(
        out_dir / "B_trigramas_TODOS.csv", index=False, encoding="utf-8-sig"
    )

    df_terminos.to_csv(
        out_dir / "comparacion_terminos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n✅ Resultados guardados en: {out_dir.resolve()}")

if __name__ == "__main__":
    main()