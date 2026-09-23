import re
from collections import Counter
from pathlib import Path
import pandas as pd

CORPUS_PATH = Path(__file__).parent / "discursos_ia_corpus.txt"

KEYWORDS = {
    "regulacion": ["regulacion", "regular", "regulacion", "gobernanza", "norma", "normas",
                   "ley", "legal", "reglas", "supervision", "derechos"],
    "innovacion": ["innovacion", "innovar", "inversion", "investigacion", "oportunidad",
                   "desarrollo", "productividad", "tecnologia"],
    "empleo": ["empleo", "trabajo", "trabajador", "trabajadores", "laboral", "ocupacion",
               "tarea", "capacitacion", "competencias", "automatizacion"],
    "riesgo_seguridad": ["riesgo", "seguridad", "seguro", "amenaza", "peligro",
                         "vulnerabilidad", "ciberseguridad", "proteccion", "sesgo"],
}

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
            print("⚠️ No se pudo reconocer este bloque:")
            print(bloque[:200])
            print()
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
STOPWORDS = {
    "el","la","los","las","un","una","unos","unas","y","o","e","u","de","del",
    "al","a","ante","bajo","con","contra","desde","en","entre","hacia","hasta",
    "para","por","sin","sobre","tras","que","se","su","sus","es","son","ser",
    "ha","han","hay","como","más","muy","también","esta","este","estas","estos",
    "esa","ese","esas","esos","lo","le","les","ya","no","pero","sino","cuando",
    "donde","quienes","quien","cuyo","cuya","cada","puede","pueden","debe","deben"
}

def tokenizar(texto: str) -> list:
    palabras = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    return [p for p in palabras if p not in STOPWORDS]

def normalizar_palabra(palabra: str) -> str:
    # Permite que las palabras clave funcionen aunque el texto tenga tildes.
    reemplazos = str.maketrans("áéíóúü", "aeiouu")
    return palabra.translate(reemplazos)

def construir_ngramas(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tabla_frecuencia_ngramas(tokens: list, n: int, top: int | None = 20) -> pd.DataFrame:
    """
    top=None (o cualquier valor <= 0) devuelve TODOS los n-gramas encontrados,
    no solo los más frecuentes. Counter.most_common(None) ya devuelve todo
    ordenado de mayor a menor, así que basta con no pasarle un número.
    """
    grams = construir_ngramas(tokens, n)
    total = len(grams)
    counts = Counter(grams)
    limite = None if (top is None or top <= 0) else top
    filas = [{
        "ngrama": " ".join(g),
        "frecuencia": c,
        "frecuencia_relativa": round(c/total, 5) if total else 0
    } for g, c in counts.most_common(limite)]
    return pd.DataFrame(filas)

def ngramas_con_palabra_clave(tokens: list, n: int, palabra: str, top: int = 5) -> pd.DataFrame:
    grams = construir_ngramas(tokens, n)
    palabra_norm = normalizar_palabra(palabra)
    filtrados = [g for g in grams if any(normalizar_palabra(x) == palabra_norm for x in g)]
    total = len(grams)
    counts = Counter(filtrados)
    filas = [{
        "ngrama": " ".join(g),
        "frecuencia": c,
        "frecuencia_relativa": round(c/total, 5) if total else 0
    } for g, c in counts.most_common(top)]
    return pd.DataFrame(filas)

def palabras_distintivas(tokens_a: list, tokens_b: list, top: int = 15) -> pd.DataFrame:
    freq_a, freq_b = Counter(tokens_a), Counter(tokens_b)
    total_a, total_b = len(tokens_a), len(tokens_b)
    vocab = set(freq_a) | set(freq_b)
    filas = []
    for palabra in vocab:
        rel_a = freq_a.get(palabra, 0) / total_a if total_a else 0
        rel_b = freq_b.get(palabra, 0) / total_b if total_b else 0
        filas.append({
            "palabra": palabra,
            "frec_rel_A": round(rel_a, 6),
            "frec_rel_B": round(rel_b, 6),
            "diferencia": round(rel_a-rel_b, 6)
        })
    df = pd.DataFrame(filas).sort_values("diferencia", ascending=False)
    return pd.concat([df.head(top), df.tail(top)]).reset_index(drop=True)

def main():
    documentos = cargar_corpus(CORPUS_PATH)
    print(f"Documentos cargados: {len(documentos)}")
    for d in documentos:
        print(f"  - [{d['tema']:16s}] {d['titulo']}")

    for d in documentos:
        d["tokens"] = tokenizar(d["texto"])

    todos_los_tokens = [tok for d in documentos for tok in d["tokens"]]
    print(f"\nTotal de tokens: {len(todos_los_tokens)}")
    print(f"Vocabulario unico: {len(set(todos_los_tokens))}")

    for n, nombre in [(1,"UNIGRAMAS"),(2,"BIGRAMAS"),(3,"TRIGRAMAS")]:
        print("\n" + "="*70)
        print(f"20 {nombre} MAS FRECUENTES")
        print("="*70)
        print(tabla_frecuencia_ngramas(todos_los_tokens, n, 20).to_string(index=False))

    print("\n" + "="*70)
    print("N-GRAMAS ASOCIADOS A CADA TEMA (BIGRAMAS)")
    print("="*70)
    for tema, palabras in KEYWORDS.items():
        print(f"\n--- {tema} ---")
        for palabra in palabras:
            tabla = ngramas_con_palabra_clave(todos_los_tokens, 2, palabra, 5)
            if not tabla.empty:
                print(f"  '{palabra}':")
                for _, fila in tabla.iterrows():
                    print(f"      {fila['ngrama']:40s} freq={fila['frecuencia']} rel={fila['frecuencia_relativa']}")

    tokens_empleo = [t for d in documentos if d["tema"] == "empleo" for t in d["tokens"]]
    tokens_riesgo = [t for d in documentos if d["tema"] == "riesgo" for t in d["tokens"]]
    tokens_innovacion = [t for d in documentos if d["tema"] == "innovacion" for t in d["tokens"]]

    if tokens_empleo and tokens_riesgo:
        print("\n" + "="*70)
        print("PALABRAS DISTINTIVAS: empleo (A) vs. riesgo (B)")
        print("="*70)
        print(palabras_distintivas(tokens_empleo, tokens_riesgo, 10).to_string(index=False))

    if tokens_innovacion and tokens_riesgo:
        print("\n" + "="*70)
        print("PALABRAS DISTINTIVAS: innovacion (A) vs. riesgo (B)")
        print("="*70)
        print(palabras_distintivas(tokens_innovacion, tokens_riesgo, 10).to_string(index=False))

    print("\n" + "="*70)
    print("APROXIMACION: ¿QUE VISION DE LA IA PREDOMINA?")
    print("="*70)

    bigramas_totales = construir_ngramas(todos_los_tokens, 2)
    conteo_bigramas = Counter(bigramas_totales)
    conteo_por_tema = {tema: 0 for tema in KEYWORDS}

    for gram, freq in conteo_bigramas.most_common(150):
        texto_gram = " ".join(normalizar_palabra(x) for x in gram)
        for tema, palabras in KEYWORDS.items():
            if any(normalizar_palabra(p) in texto_gram.split() for p in palabras):
                conteo_por_tema[tema] += freq

    resumen = pd.DataFrame(
        [{"tema": t, "frecuencia_acumulada": c} for t,c in conteo_por_tema.items()]
    ).sort_values("frecuencia_acumulada", ascending=False)
    print(resumen.to_string(index=False))

    out_dir = Path("resultados")
    out_dir.mkdir(exist_ok=True)
    # top=None => TODOS los n-gramas del corpus, sin cortar en 50.
    tabla_frecuencia_ngramas(todos_los_tokens,1,None).to_csv(out_dir/"unigramas_TODOS.csv",index=False,encoding="utf-8-sig")
    tabla_frecuencia_ngramas(todos_los_tokens,2,None).to_csv(out_dir/"bigramas_TODOS.csv",index=False,encoding="utf-8-sig")
    tabla_frecuencia_ngramas(todos_los_tokens,3,None).to_csv(out_dir/"trigramas_TODOS.csv",index=False,encoding="utf-8-sig")
    resumen.to_csv(out_dir/"resumen_por_tema.csv",index=False,encoding="utf-8-sig")
    print(f"\nTablas guardadas en: {out_dir.resolve()}")
    print(f"  unigramas_TODOS.csv: {len(set(todos_los_tokens))} filas (vocabulario único)")
    print(f"  bigramas_TODOS.csv:  {len(set(construir_ngramas(todos_los_tokens,2)))} filas")
    print(f"  trigramas_TODOS.csv: {len(set(construir_ngramas(todos_los_tokens,3)))} filas")

if __name__ == "__main__":
    main()