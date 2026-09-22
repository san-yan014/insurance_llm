import sys
import json
import gzip
import pandas as pd

EXCEL_CELL_LIMIT = 32767

def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def to_dataframe(data):
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                rows.append({"id": key, **value})
            else:
                rows.append({"id": key, "value": value})
        return pd.DataFrame(rows)
    if isinstance(data, list):
        return pd.json_normalize(data)
    raise ValueError("expected a JSON list or object at the top level")


def truncate_long_cells(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        too_long = df[col].astype(str).str.len() > EXCEL_CELL_LIMIT
        if too_long.any():
            df.loc[too_long, col] = df.loc[too_long, col].astype(str).str.slice(0, EXCEL_CELL_LIMIT - 20) + " [TRUNCATED]"
    return df


def main():
    if len(sys.argv) < 2:
        print("usage: python json_to_excel.py <input.json[.gz]> [output.xlsx]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.rsplit(".json", 1)[0] + ".xlsx"

    data = load_json(input_path)
    df = to_dataframe(data)
    df = truncate_long_cells(df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="articles", index=False)
        ws = writer.sheets["articles"]
        for idx, col in enumerate(df.columns, start=1):
            width = 60 if df[col].astype(str).str.len().mean() > 40 else 20
            ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = width

    print(f"{len(df)} rows written to {output_path}")


if __name__ == "__main__":
    main()