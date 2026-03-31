import os, sys
import warnings
import pandas as pd
from Bio.PDB import PDBList, PPBuilder, MMCIFParser

from pathlib import Path

if "PROJECT_ROOT" in os.environ:
    root_path = Path(os.environ["PROJECT_ROOT"]).resolve()
else:
    # fallback: assume this file is somewhere inside src/
    root_path = Path(__file__).resolve().parents[1]
    
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import src.config as config

warnings.filterwarnings("ignore")

os.makedirs(config.PDB_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR, exist_ok=True)


def download_pdb_files(pdb_ids, pdb_dir):
    pdbl = PDBList()
    for pdb_id in pdb_ids:
        cif_file = find_cif_file(pdb_id, pdb_dir)
        if cif_file is not None:
            print(f"Already have {pdb_id}, skipping download")
            continue
        pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir)
        print(f"Downloaded {pdb_id}")


def find_cif_file(pdb_id, pdb_dir):
    pdb_id_lower = pdb_id.lower()
    for fname in os.listdir(pdb_dir):
        if pdb_id_lower in fname.lower() and fname.endswith(".cif"):
            return os.path.join(pdb_dir, fname)
    return None


def extract_phi_psi(pdb_ids, pdb_dir):
    parser = MMCIFParser(QUIET=True)
    builder = PPBuilder()
    records = []

    for pdb_id in pdb_ids:
        cif_file = find_cif_file(pdb_id, pdb_dir)

        if cif_file is None:
            print(f"Could not find file for {pdb_id}, skipping")
            continue

        structure = parser.get_structure(pdb_id, cif_file)
        polypeptides = builder.build_peptides(structure)

        count_before = len(records)
        for poly in polypeptides:
            phi_psi = poly.get_phi_psi_list()
            for phi, psi in phi_psi:
                if phi is not None and psi is not None:
                    records.append({
                        "pdb_id": pdb_id,
                        "phi": round(phi * (180.0 / 3.141592653589793), 4),
                        "psi": round(psi * (180.0 / 3.141592653589793), 4)
                    })

        print(f"Extracted phi/psi from {pdb_id}: {len(records) - count_before} residues")

    return records


def build_dataset(pdb_ids, output_csv):
    print(f"\nDownloading {len(pdb_ids)} structures...")
    download_pdb_files(pdb_ids, config.PDB_DIR)

    print("Extracting phi/psi angles...")
    records = extract_phi_psi(pdb_ids, config.PDB_DIR)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Done. {len(df)} residues from {df['pdb_id'].nunique()} structures -> {output_csv}")
    print(df.describe())
    return df


def main():
    print("=== Building original dataset ===")
    build_dataset(config.ORIGINAL_PDB_IDS, config.CSV_ORIGINAL)

    print("\n=== Building augmented dataset ===")
    build_dataset(config.AUGMENTED_PDB_IDS, config.CSV_AUGMENTED)


if __name__ == "__main__":
    main()
