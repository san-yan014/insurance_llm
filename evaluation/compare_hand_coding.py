import sys
import pandas as pd

SUB_MAP = {"a": "positive", "b": "negative", "c": "neutral"}


def map_sentiment(sub):
    if pd.isna(sub):
        return None
    text = str(sub).strip().lower()
    if text.startswith("positive"):
        return "positive"
    if text.startswith("negative"):
        return "negative"
    if text.startswith("neutral"):
        return "neutral"
    return SUB_MAP.get(text[:1])


def base_topic(label):
    if pd.isna(label):
        return None
    return label.split(" (")[0].strip()


def match_key(label):
    base = base_topic(label)
    if base == "market conditions":
        return base
    return label


def reconstruct_topic(row):
    if pd.notna(row["agent-client relationship"]) or pd.notna(row["relationship_sub"]):
        edu = pd.notna(row["client_education"])
        sub = row["relationship_sub"]
        mod = map_sentiment(sub)
        if mod and edu:
            return f"agent-client relationship ({mod} + client education)"
        if mod:
            return f"agent-client relationship ({mod})"
        if edu:
            return "agent-client relationship (client education)"
        return "agent-client relationship"

    if pd.notna(row["market conditions"]) or pd.notna(row["negative_relationship (market conditions)"]):
        neg = pd.notna(row["negative_relationship (market conditions)"])
        return "market conditions (negative relationship)" if neg else "market conditions"

    if pd.notna(row["technology/automation"]):
        return "technology/automation"
    if pd.notna(row["consolidation"]):
        return "consolidation"
    if pd.notna(row["family/community"]):
        return "family/community"

    if pd.notna(row["legal/regulatory"]):
        edu = pd.notna(row["client_education"])
        return "legal/regulatory (client education)" if edu else "legal/regulatory"

    if pd.notna(row["client_education"]):
        return "client_education"

    return None


def align(hand, llm):
    hand = hand.copy()
    llm = llm.copy()
    hand["ord"] = hand.groupby("article_key").cumcount()
    llm["ord"] = llm.groupby("article_key").cumcount()
    return hand.merge(llm, on=["article_key", "ord"], suffixes=("_hand", "_llm"), how="inner")


MANUAL_OVERRIDES = {
    ("will", "2016-11-18_Vol279_No16.pdf|AAMGA and NAPSLO Explore Merger", "Levy and Leonard lauded"): "consolidation",
}


def apply_override(coder, article_key, relevant_text):
    for (c, key, snippet), code in MANUAL_OVERRIDES.items():
        if c == coder and key == article_key and relevant_text.startswith(snippet):
            return code
    return None


def build_rows(hand, llm, coder):
    merged = align(hand, llm)
    rows = []
    for _, r in merged.iterrows():
        base = dict(coder=coder, article_key=r["article_key"], title=r["title_hand"], relevant_text=r["relevant_text_hand"])
        override = apply_override(coder, r["article_key"], r["relevant_text_hand"])
        if override is not None:
            rows.append({**base, "our_code": override, "llm_code": r["topic"]})
            continue
        topic = reconstruct_topic(r)
        if topic is not None:
            rows.append({**base, "our_code": topic, "llm_code": r["topic"]})
    return pd.DataFrame(rows)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def weighted_f1(y_true, y_pred):
    labels = set(y_true.dropna()) | set(y_pred.dropna())
    weighted_sum = 0.0
    weight_total = 0
    for label in labels:
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        weight = (y_true == label).sum()
        weighted_sum += f1 * weight
        weight_total += weight
    return weighted_sum / weight_total if weight_total else 0.0


def main():
    if len(sys.argv) < 2:
        print("usage: python compare_hand_coding.py <workbook.xlsx> [output.xlsx]")
        sys.exit(1)

    workbook = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else workbook.rsplit(".", 1)[0] + "_comparison.xlsx"

    xl = pd.ExcelFile(workbook)
    edna = build_rows(xl.parse("edna's 50"), xl.parse("edna's answers"), "edna")
    will = build_rows(xl.parse("will's 50"), xl.parse("will's answers"), "will")
    helen = build_rows(xl.parse("helen's 50"), xl.parse("helen's answers"), "helen")
    combined = pd.concat([edna, will, helen], ignore_index=True)

    for name, df in [("edna", edna), ("will", will), ("helen", helen), ("combined", combined)]:
        df["match"] = df["our_code"].map(match_key) == df["llm_code"].map(match_key)
        acc = accuracy(df["our_code"].map(match_key), df["llm_code"].map(match_key))
        f1 = weighted_f1(df["our_code"].map(match_key), df["llm_code"].map(match_key))
        n_match = df["match"].sum()
        n_mismatch = (~df["match"]).sum()
        print(f"{name}: {n_match} matches out of {len(df)} ({acc:.1%}), F1 score = {f1:.3f}")
        print(f"  True: {n_match}, False: {n_mismatch}")

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="combined", index=False)

    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()