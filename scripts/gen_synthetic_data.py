# scripts/gen_synthetic_data.py
import os
import numpy as np
import pandas as pd
from tqdm import trange
import argparse

def gen_data(out_dir="data", num_patients=50, min_scans=1, max_scans=5, spec_len=512, num_tab_features=10, random_seed=42):
    np.random.seed(random_seed)
    os.makedirs(out_dir, exist_ok=True)

    spectra_rows = []
    clinic_rows = {}

    for pid_idx in range(num_patients):
        patient_id = f"P{1000+pid_idx}"   # e.g. P1000, P1001...
        n_scans = np.random.randint(min_scans, max_scans+1)
        # randomly assign label: ~balanced
        label = "DM" if np.random.rand() < 0.5 else "CTRL"

        # clinical features (one row per patient). We'll include Group column too.
        tab = np.random.randn(num_tab_features).astype(float)
        clinic_rows[patient_id] = list(tab) + [label]

        for scan_idx in range(n_scans):
            # create a synthetic spectrum: combination of random peaks + noise
            x = np.linspace(0, 1, spec_len)
            # random few gaussian peaks
            spec = np.zeros_like(x)
            n_peaks = np.random.randint(1, 5)
            for _ in range(n_peaks):
                mu = np.random.rand()
                sigma = 0.005 + 0.02 * np.random.rand()
                amp = (0.5 + np.random.rand()) * (1.0 if label=="DM" else 0.8)  # tiny signal diff by label
                spec += amp * np.exp(-0.5 * ((x - mu) / sigma)**2)
            # add low frequency baseline + noise
            baseline = 0.1 * np.sin(2*np.pi*5*x) + 0.05*np.random.randn(spec_len)
            spec = spec + baseline + 0.02*np.random.randn(spec_len)

            row = {}
            row['Sample'] = f"{patient_id}-{scan_idx}"
            row['Group'] = label
            # wave columns w0...w{spec_len-1}
            for i,val in enumerate(spec):
                row[f"w{i}"] = float(val)
            spectra_rows.append(row)

    # build spectra dataframe
    spectra_df = pd.DataFrame(spectra_rows)
    spectra_csv = os.path.join(out_dir, "spectra.csv")
    spectra_df.to_csv(spectra_csv, index=False, float_format="%.6f")

    # build clinical dataframe: index is patient_id, columns feat0..featN, plus Group
    col_names = [f"feat{i}" for i in range(num_tab_features)] + ["Group"]
    clinic_df = pd.DataFrame.from_dict(clinic_rows, orient="index", columns=col_names)
    clinic_df.index.name = "PatientID"
    clinical_csv = os.path.join(out_dir, "clinical.csv")
    clinic_df.to_csv(clinical_csv, index=True, float_format="%.6f")

    print(f"Generated {len(spectra_df)} spectra rows for {num_patients} patients.")
    print("spectra csv:", spectra_csv)
    print("clinical csv:", clinical_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--num_patients", type=int, default=50)
    parser.add_argument("--min_scans", type=int, default=1)
    parser.add_argument("--max_scans", type=int, default=5)
    parser.add_argument("--spec_len", type=int, default=512)
    parser.add_argument("--num_tab_features", type=int, default=10)
    args = parser.parse_args()

    gen_data(out_dir=args.out_dir,
             num_patients=args.num_patients,
             min_scans=args.min_scans,
             max_scans=args.max_scans,
             spec_len=args.spec_len,
             num_tab_features=args.num_tab_features)
