"""Embed CATH domains with a pretrained PST model, one vector per domain.

A domain's vector is the mean of PST's per-residue representations. With ``--include-seq`` (only
for the ``*_so`` models, trained with structure-only updates) the structure-aware and the
sequence-only representations are averaged, as in PST's own ``scripts/pst_extract.py``. The run
then also saves the two parts on their own: the sequence-only one is plain ESM-2, because a ``*_so``
model keeps the ESM-2 weights, so the three variants compare structure with sequence alone.

    python -m scripts.extract_embeddings --datadir data --cath data/cath_data.csv --model pst_t33_so --include-seq

Restored from the first version of this repository (``scripts/protein_domain_extract.py``).
The output file (default ``data/embeddings/<model>.pt``) holds a dict with ``embeddings``
(N x D float32), ``labels`` (N,), and ``cath_id`` and ``superfamily`` (lists of N strings); with
``--include-seq`` also ``embeddings_structure`` and ``embeddings_sequence``. Domains are embedded
shortest first, so that a batch holds sequences of similar length and little of it is padding.
"""

import argparse
from pathlib import Path

import torch
from pst.esm2 import PST
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from scripts.cath_dataset import CathDomainDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datadir", default="data", help="directory with pdb/<cath_id>.pdb; graphs are cached here")
    parser.add_argument("--cath", default="data/cath_data.csv", help="CSV with the CATH labels of the domains")
    parser.add_argument("--model", default="pst_t33_so", help="pretrained PST model, see PST/README.md")
    parser.add_argument("--include-seq", action="store_true", help="average with sequence-only representations")
    parser.add_argument("--aggr", default=None, choices=["mean", "concat"], help="combine layers (default: last layer)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--out", default=None, help="output file (default: data/embeddings/<model>.pt)")
    return parser.parse_args()


@torch.no_grad()
def embed(loader: DataLoader, model: PST, include_seq: bool, aggr: str | None, device: torch.device) -> dict:
    embeddings, structure, sequence, labels, cath_ids, superfamilies = [], [], [], [], [], []
    for data in tqdm(loader, desc="Embedding"):
        data = data.to(device)
        out = model(data, return_repr=True, aggr=aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        if include_seq:
            data.edge_index = None  # PST without the structure graph is the sequence model
            out_seq = model(data, return_repr=True, aggr=aggr)[data.idx_mask]
            structure.append(global_mean_pool(out, batch).cpu())
            sequence.append(global_mean_pool(out_seq, batch).cpu())
            out = (out + out_seq) / 2
        embeddings.append(global_mean_pool(out, batch).cpu())
        labels.append(data.y.cpu())
        cath_ids.extend(data.cath_id)
        superfamilies.extend(data.superfamily)
    result = {
        "embeddings": torch.cat(embeddings),
        "labels": torch.cat(labels),
        "cath_id": cath_ids,
        "superfamily": superfamilies,
    }
    if include_seq:
        result.update(embeddings_structure=torch.cat(structure), embeddings_sequence=torch.cat(sequence))
    return result


def main():
    args = parse_args()
    if args.include_seq and not args.model.endswith("_so"):
        raise SystemExit("--include-seq needs a model trained with structure-only updates (a *_so model)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = Path(".cache/pst") / f"{args.model}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    # PST checkpoints store their config as a pickled object, hence weights_only=False.
    model, _ = PST.from_pretrained_url(args.model, weights, map_location="cpu", weights_only=False)
    model.eval().to(device)

    dataset = CathDomainDataset(root=args.datadir, cath_csv=args.cath)
    dataset = dataset.index_select(sorted(range(len(dataset)), key=lambda i: dataset[i].num_nodes))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    result = embed(loader, model, args.include_seq, args.aggr, device)
    result.update(model=args.model, include_seq=args.include_seq, aggr=args.aggr)

    out = Path(args.out or f"data/embeddings/{args.model}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"Saved {tuple(result['embeddings'].shape)} embeddings to {out}")


if __name__ == "__main__":
    main()
