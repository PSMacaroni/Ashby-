"""
Pick a NEW set of TCGA colon/rectum diagnostic slides (patients not used anywhere yet) for a held-out test of the
analysis, and save TCGA's own labels for every COAD/READ case (for later cut-off checks, no pathologist needed).

Writes (laptop, network only, no GPU):
  <out>/manifest_<name>.csv    file_id, file_name, file_size, md5sum, case, project, pT, pN, stage, ajcc_edition,
                               prior_treatment, T, primary_diagnosis: the picked slides, in processing order
                               (T stages interleaved, so a run cut short still covers every stage)
  <out>/clinical_labels.csv    every COAD/READ case: pT, pN, stage, histology (mucinous or not), lymphatic / vascular /
                               perineural invasion, MSI scores (MANTIS, MSIsensor; cBioPortal PanCancer Atlas)

    python scripts/pick_gdc_slides.py --exclude calibration/manifest_v1.csv --n 80 --name holdout_v1 --out calibration
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GDC = "https://api.gdc.cancer.gov/files"
CBIO = "https://www.cbioportal.org/api/studies/coadread_tcga_pan_can_atlas_2018/clinical-data?clinicalDataType=SAMPLE&projection=SUMMARY"
OUR_5 = ["TCGA-AA-3875", "TCGA-CM-4746", "TCGA-AH-6547", "TCGA-AA-3844", "TCGA-A6-3809"]
DX = ["file_id", "file_name", "file_size", "md5sum", "cases.submitter_id", "cases.project.project_id"] + [
    f"cases.diagnoses.{f}" for f in ("ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_stage",
                                      "ajcc_staging_system_edition", "prior_treatment", "primary_diagnosis",
                                      "lymphatic_invasion_present", "vascular_invasion_present",
                                      "perineural_invasion_present")]


def gdc_slides():
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-COAD", "TCGA-READ"]}},
        {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
        {"op": "=", "content": {"field": "access", "value": "open"}}]}
    body = json.dumps({"filters": filters, "fields": ",".join(DX), "format": "JSON", "size": 5000}).encode()
    req = urllib.request.Request(GDC, data=body, headers={"Content-Type": "application/json"})
    hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
    rows = []
    for h in hits:
        case = h["cases"][0]
        d = (case.get("diagnoses") or [{}])[0]
        rows.append({"file_id": h["file_id"], "file_name": h["file_name"], "file_size": h["file_size"],
                     "md5sum": h["md5sum"], "case": case["submitter_id"], "project": case["project"]["project_id"],
                     "pT": d.get("ajcc_pathologic_t"), "pN": d.get("ajcc_pathologic_n"),
                     "stage": d.get("ajcc_pathologic_stage"), "ajcc_edition": d.get("ajcc_staging_system_edition"),
                     "prior_treatment": d.get("prior_treatment"), "primary_diagnosis": d.get("primary_diagnosis"),
                     "lymphatic_invasion": d.get("lymphatic_invasion_present"),
                     "vascular_invasion": d.get("vascular_invasion_present"),
                     "perineural_invasion": d.get("perineural_invasion_present")})
    return pd.DataFrame(rows)


def msi_scores():
    """MSI scores per patient from cBioPortal (TCGA PanCancer Atlas). MANTIS > 0.4 is the usual MSI-high cut-off."""
    try:
        data = json.load(urllib.request.urlopen(CBIO, timeout=120))
    except Exception as e:
        print(f"cBioPortal MSI scores not available: {e!r}")
        return pd.DataFrame(columns=["case"])
    df = pd.DataFrame([{"case": r["patientId"], "attr": r["clinicalAttributeId"], "value": r["value"]} for r in data
                       if r["clinicalAttributeId"] in ("MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE")])
    if df.empty:
        return pd.DataFrame(columns=["case"])
    df = df.pivot_table(index="case", columns="attr", values="value", aggfunc="first").reset_index()
    return df.rename(columns={"MSI_SCORE_MANTIS": "msi_mantis", "MSI_SENSOR_SCORE": "msi_sensor"})


def t_group(pt):
    if not isinstance(pt, str):
        return None
    if pt.startswith("Tis"):
        return "T1"
    m = re.match(r"T(\d)", pt)
    return f"T{m.group(1)}" if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", nargs="*", default=[], help="manifests whose patients are already used")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--name", default="holdout_v1")
    ap.add_argument("--out", default="calibration")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)

    df = gdc_slides()
    df["T"] = df.pT.map(t_group)
    msi = msi_scores()
    labels = (df.sort_values("file_name").drop_duplicates("case")
              [["case", "project", "pT", "pN", "stage", "ajcc_edition", "prior_treatment", "primary_diagnosis",
                "lymphatic_invasion", "vascular_invasion", "perineural_invasion"]].merge(msi, on="case", how="left"))
    labels["mucinous"] = labels.primary_diagnosis.fillna("").str.contains("ucinous")
    labels.to_csv(out / "clinical_labels.csv", index=False)
    print(f"clinical_labels.csv: {len(labels)} cases, {int(labels.mucinous.sum())} mucinous, "
          f"{int(labels.get('msi_mantis', pd.Series(dtype=str)).notna().sum())} with an MSI score")

    used = set(OUR_5)
    for m in args.exclude:
        used |= set(pd.read_csv(m).case)
    pool = df[~df.case.isin(used) & df["T"].notna() & (df.prior_treatment.fillna("No") != "Yes")
              & df.file_name.str.contains("-01Z-00-DX")]
    pool = pool.sort_values("file_name").drop_duplicates("case")      # one slide per patient (DX1 first)
    print("available, new patients:", pool["T"].value_counts().sort_index().to_dict())

    rng = np.random.default_rng(args.seed)
    groups = {t: g.iloc[rng.permutation(len(g))] for t, g in pool.groupby("T")}
    quota, left = {}, args.n                                          # as even as the rare stages allow
    for t in sorted(groups, key=lambda t: len(groups[t])):
        share = left // (len(groups) - len(quota))
        quota[t] = min(share, len(groups[t]))
        left -= quota[t]
    picked = [groups[t].iloc[:quota[t]] for t in sorted(groups)]
    order = []                                                        # interleave stages: T1, T2, T3, T4, T1, ...
    for i in range(max(len(p) for p in picked)):
        order += [p.iloc[[i]] for p in picked if i < len(p)]
    man = pd.concat(order, ignore_index=True)
    cols = ["file_id", "file_name", "file_size", "md5sum", "case", "project", "pT", "pN", "stage", "ajcc_edition",
            "prior_treatment", "T", "primary_diagnosis"]
    man[cols].to_csv(out / f"manifest_{args.name}.csv", index=False)
    print(f"manifest_{args.name}.csv: {len(man)} slides {man['T'].value_counts().sort_index().to_dict()}, "
          f"{man.file_size.sum() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
