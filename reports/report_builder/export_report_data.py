"""Collect every pipeline result into one JSON file for build_report.js.

    python reports/report_builder/export_report_data.py

Nothing is computed here except table reshaping, so every number in the
report comes straight from reports/results (and the model card).
"""
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def clean(o):
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def main():
    data = {"tables": {}, "json": {}, "figures": {}}
    for p in sorted(RES.glob("*.csv")):
        df = pd.read_csv(p)
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "index"})
        data["tables"][p.stem] = clean(df.to_dict(orient="records"))
    for p in sorted(RES.glob("*.json")):
        data["json"][p.stem] = clean(json.loads(p.read_text()))
    card = ROOT / "models" / "best_model" / "model_card.json"
    data["json"]["model_card"] = clean(json.loads(card.read_text()))
    data["tables"]["data_dictionary"] = clean(pd.read_csv(ROOT / "data" / "processed" / "data_dictionary.csv")
                                              .to_dict(orient="records"))
    for p in sorted(FIG.glob("*.png")):
        w, h = png_size(p)
        data["figures"][p.stem] = {"path": str(p), "width": w, "height": h}

    # one synthetic person per most-likely mode, to show individual-level output
    pop = pd.read_csv(ROOT / "data" / "synthetic" / "synthetic_population.csv")
    pred = pd.read_csv(ROOT / "data" / "synthetic" / "synthetic_predictions.csv")
    both = pop.merge(pred, on="ID")
    av_cols = ["Walk_AV", "TwoW_AV", "Auto_AV", "Car_AV", "Bus_AV", "LocalTrain_AV"]
    names = ["Walk", "2W", "Auto", "Car", "Bus", "Rail"]
    picks = both.groupby("predicted_mode", sort=False).head(1)
    rows = []
    for _, r in picks.iterrows():
        rows.append({"ID": int(r["ID"]), "distance_km": r["Distance_km"], "income": int(r["Income"]),
                     "cars": int(r["Car_Ownership"]),
                     "available": ", ".join(n for n, c in zip(names, av_cols) if r[c] == 1),
                     **{n: round(float(r[f"P_{m}"]), 3) for n, m in
                        zip(names, ["walk", "twow", "auto", "car", "bus", "train"])},
                     "most_likely": r["predicted_mode"]})
    data["tables"]["example_individuals"] = rows

    # sizes quoted in the text
    import sys
    sys.path.insert(0, str(ROOT))
    from src.feature_engineering import CATEGORICAL, NUMERIC, FeatureBuilder
    from src.preprocessing import make_ml_preprocessor
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    prep = make_ml_preprocessor(CATEGORICAL, NUMERIC).fit(FeatureBuilder().transform(train))
    gaps = {}
    for v in ["Income", "Age", "Education", "Occup", "Gender", "Area_Type", "Dwelling", "HH_Size", "Children",
              "Working_Adults", "Car_Ownership", "TwoW_Ownership", "ParkingSpace", "Car_AV", "TwoW_AV",
              "Trip_Purpose", "SAPT", "Walk_AV", "Auto_AV", "Bus_AV"]:
        a, b = train[v].value_counts(normalize=True), pop[v].value_counts(normalize=True)
        idx = a.index.union(b.index)
        gaps[v] = float(100 * (a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).abs().max())
    worst = max(gaps, key=gaps.get)
    data["json"]["sizes"] = {
        "long_rows": int(len(pd.read_csv(ROOT / "data" / "processed" / "train_long_format.csv"))),
        "n_ml_features": int(len(prep.get_feature_names_out())),
        "n_categorical": len(CATEGORICAL), "n_numeric": len(NUMERIC),
        "bus_av_train_pct": float(100 * train["Bus_AV"].mean()), "bus_av_synthetic_pct": float(100 * pop["Bus_AV"].mean()),
        "largest_category_gap": {"variable": worst, "pp": gaps[worst]},
        "second_largest_gap_pp": sorted(gaps.values())[-2],
    }

    junit = RES / "test_report.xml"
    if junit.exists():
        import xml.etree.ElementTree as ET
        root = ET.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        cases = []
        for c in suite.iter("testcase"):
            status = "passed"
            for tag in ("failure", "error", "skipped"):
                if c.find(tag) is not None:
                    status = tag
            cases.append({"file": c.get("classname", "").split(".")[-1], "test": c.get("name"),
                          "seconds": float(c.get("time", 0)), "status": status})
        data["json"]["tests"] = {k: suite.get(k) for k in ("tests", "failures", "errors", "skipped", "time")}
        data["tables"]["tests"] = cases
    out = Path(__file__).with_name("report_data.json")
    out.write_text(json.dumps(data, indent=1, default=str))
    print(f"wrote {out} ({len(data['tables'])} tables, {len(data['figures'])} figures)")


if __name__ == "__main__":
    main()
