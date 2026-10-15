"""
Reads the 'handcoded vs llm coded' sheet and produces a per-code F1 analysis.
For each code (market conditions modifiers collapsed into one), computes
TP/FP/FN/TN, Precision, Recall, F1, Cohen's Kappa, and agreement stats as
live Excel formulas. Outputs one sheet per coder plus a combined sheet,
each with an OVERALL (accuracy) row and a Weighted avg F1 row on top.

Input: 
    Side by side comparison between manual human codings and LLM codings clearly defined by words 
    
    e.g. 

    our_code	                            llm_code
    agent-client            relationship (neutral)	technology/automation
    agent-client            relationship (neutral)	agent-client relationship (neutral)

Output: 

    Builds a per-code F1/Kappa breakdown (one sheet per coder + combined) from the handcoded-vs-LLM comparison sheet.

Usage: python f1_analysis.py 150 standard articles (handcoded vs llm coded).xlsx

"""
import re
import sys
import pandas as pd
from openpyxl.utils import get_column_letter
 
SHEET_NAME = "handcoded vs llm coded"
 
COLUMNS = [
    "Code", "LLM Codes", "{h} Codes", "TP", "FP", "FN", "TN",
    "Precision", "Recall", "F1", "Cohen's Kappa",
    "LLM %", "{h} %", "Gap (LLM% - Human%)",
    "FP Rate (FP / LLM Codes)", "FN Rate (FN / Human Codes)",
    "Po (Observed Agreement)", "Pe (Expected Agreement by Chance)",
]
 
 
def collapse_key(label):
    if pd.isna(label):
        return None
    # drop "client education" add-on, e.g. "(neutral + client education)" -> "(neutral)"
    label = re.sub(r"\s*\+?\s*client education", "", label)
    label = re.sub(r"\s*\(\s*\)", "", label).strip()  # remove empty "()" if left over
    base = label.split(" (")[0].strip()
    if base == "market conditions":
        return base
    return label
 
 
def raw_counts(df, human_label):
    n = len(df)
    our = df["our_code"].map(collapse_key)
    llm = df["llm_code"].map(collapse_key)
    codes = sorted(set(our.dropna()) | set(llm.dropna()))
    rows = []
    for code in codes:
        llm_yes = llm == code
        human_yes = our == code
        tp = int((llm_yes & human_yes).sum())
        fp = int((llm_yes & ~human_yes).sum())
        fn = int((~llm_yes & human_yes).sum())
        tn = n - tp - fp - fn
        rows.append({
            "Code": code,
            "LLM Codes": int(llm_yes.sum()),
            f"{human_label} Codes": int(human_yes.sum()),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows).sort_values("LLM Codes", ascending=False).reset_index(drop=True)
 
 
def overall_rows(counts, human_label):
    n = int(counts[["TP", "FP", "FN", "TN"]].iloc[0].sum())
    return pd.DataFrame([
        {"Code": "OVERALL / Micro F1", "LLM Codes": n, f"{human_label} Codes": n,
         "TP": int(counts["TP"].sum()), "FP": int(counts["FP"].sum()), "FN": int(counts["FN"].sum())},
        {"Code": "Macro F1 (unweighted mean of per-code F1)"},
        {"Code": "Weighted F1 (weighted by Human Codes)"},
        {"Code": "Overall Cohen's Kappa"},
    ])
 
 
def write_formulas(ws, num_codes):
    first = 6  # codes start at row 6 (rows 2-5 are the summary rows)
    last = 5 + num_codes
    for r in range(first, last + 1):
        ws[f"H{r}"] = f"=IFERROR(D{r}/(D{r}+E{r}),0)"
        ws[f"I{r}"] = f"=IFERROR(D{r}/(D{r}+F{r}),0)"
        ws[f"J{r}"] = f"=IFERROR(2*H{r}*I{r}/(H{r}+I{r}),0)"
        ws[f"L{r}"] = f"=B{r}/(D{r}+E{r}+F{r}+G{r})"
        ws[f"M{r}"] = f"=C{r}/(D{r}+E{r}+F{r}+G{r})"
        ws[f"N{r}"] = f"=L{r}-M{r}"
        ws[f"O{r}"] = f"=IFERROR(E{r}/B{r},0)"
        ws[f"P{r}"] = f"=IFERROR(F{r}/C{r},0)"
        ws[f"Q{r}"] = f"=(D{r}+G{r})/(D{r}+E{r}+F{r}+G{r})"
        ws[f"R{r}"] = f"=L{r}*M{r}+(1-L{r})*(1-M{r})"
        ws[f"K{r}"] = f"=IFERROR((Q{r}-R{r})/(1-R{r}),0)"
    # row 2: micro F1 (pooled TP/FP/FN)
    ws["H2"] = "=IFERROR(D2/(D2+E2),0)"
    ws["I2"] = "=IFERROR(D2/(D2+F2),0)"
    ws["J2"] = "=IFERROR(2*H2*I2/(H2+I2),0)"
    # row 3: macro F1 (unweighted mean; codes used by only one side count as F1 = 0)
    ws["J3"] = f"=AVERAGE(J{first}:J{last})"
    # row 4: weighted F1 (weighted by human support)
    ws["J4"] = f"=SUMPRODUCT(J{first}:J{last},C{first}:C{last})/SUM(C{first}:C{last})"
    # row 5: overall multi-class kappa. Po = accuracy, Pe = sum of (LLM share x human share)
    ws["Q5"] = f"=SUM(D{first}:D{last})/B2"
    ws["R5"] = f"=SUMPRODUCT(B{first}:B{last},C{first}:C{last})/B2^2"
    ws["K5"] = "=IFERROR((Q5-R5)/(1-R5),0)"
 
 
def format_sheet(ws, num_rows, human_label):
    pct_letters = ["H", "I", "J", "L", "M", "N", "O", "P", "Q", "R"]
    for letter in pct_letters:
        for row in range(2, num_rows + 1):
            ws[f"{letter}{row}"].number_format = "0.0%"
    for row in range(2, num_rows + 1):
        ws[f"K{row}"].number_format = "0.000"
    for idx, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = max(12, len(col.replace("{h}", human_label)) + 2)
 
 
def build_sheet(writer, sheet_name, df, human_label):
    counts = raw_counts(df, human_label)
    full = pd.concat([overall_rows(counts, human_label), counts], ignore_index=True)
    columns = [c.replace("{h}", human_label) for c in COLUMNS[:7]]
    full = full.reindex(columns=columns)
    full.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]
    for idx, col in enumerate(COLUMNS[7:], start=8):
        ws.cell(row=1, column=idx, value=col.replace("{h}", "Human"))
    write_formulas(ws, len(counts))
    format_sheet(ws, len(full) + 1, human_label)
 
 
def main():
    if len(sys.argv) < 2:
        print("usage: python f1_analysis.py <workbook.xlsx> [output.xlsx]")
        sys.exit(1)
 
    workbook = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else workbook.rsplit(".", 1)[0] + "_f1_analysis.xlsx"
 
    df = pd.read_excel(workbook, sheet_name=SHEET_NAME)
    # drop other_theme rows entirely (all coders, so combined matches)
    df = df[(df["our_code"] != "other_theme") & (df["llm_code"] != "other_theme")]
 
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for coder in sorted(df["coder"].dropna().unique()):
            build_sheet(writer, coder, df[df["coder"] == coder], coder.capitalize())
        build_sheet(writer, "combined", df, "Human")
 
    print(f"saved to {output}")
 
 
if __name__ == "__main__":
    main()
 