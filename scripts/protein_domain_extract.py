import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

from pst.esm2 import PST
from protein_domain_dataset import ProteinDomainDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use PST to extract per-token representations \
        for pdb files stored in datadir/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cath",
        type=str,
        required=True,
        help="Path to the CSV file containing CATH data",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data/",
        help="Path to the dataset, pdb files should be stored in data/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pst_t6",
        help="Name of pretrained PST model to download",
    )
    parser.add_argument(
        "--include-seq",
        action='store_true',
        help="Add sequence representation to the final representation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the data loader"
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default=None,
        help="How to aggregate protein representations across layers. \
        `None`: last layer; `mean`: mean pooling, `concat`: concatenation",
    )
    cfg = parser.parse_args()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

@torch.no_grad()
def compute_repr(data_loader, model, cfg):
    '''
    Calculate PST-embeddings
    '''
    embeddings = []
    labels_cath = []
    superfamilies = []
    
    for batch_idx, data in enumerate(tqdm(data_loader, desc="Processing batches")):
        data = data.to(cfg.device)
        out = model(data, return_repr=True, aggr=cfg.aggr)
        out, batch = out[data.idx_mask], data.batch[data.idx_mask]
        
        if cfg.include_seq:
            if "so" not in cfg.model:
                raise ValueError("Use models pretrained using struct only updates strategy!")
            data.edge_index = None
            out_seq = model(data, return_repr=True, aggr=cfg.aggr)
            out_seq = out_seq[data.idx_mask]
            out = (out + out_seq) * 0.5
        
        graph_embeddings = global_mean_pool(out, batch)
        
        embeddings.extend(graph_embeddings.cpu())
        labels_cath.extend(data.cath_label.cpu().numpy())
        superfamilies.extend(data.superfamilies.cpu().numpy())  # Make sure this matches the attribute name
    
    embeddings = torch.stack(embeddings).cpu().numpy()
    labels_cath = np.array(labels_cath)
    superfamilies = np.array(superfamilies)

    # Prints for debugging
    example_embedding = embeddings[0]
    print(f"Example embedding: {example_embedding}")
    print(f"Shape of example embedding: {example_embedding.shape}")
    print(f"Size of example embedding: {example_embedding.nbytes} bytes")

    print(f"Total size of embeddings array in memory: {embeddings.nbytes / 1e6} MB")

    return embeddings, labels_cath, superfamilies



def main():
    cfg = parse_args()

    cath_data = pd.read_csv(cfg.cath)

    pretrained_path = Path(f".cache/pst/{cfg.model}.pt")
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model, pretrained_path
        )
    except:
        model, model_cfg = PST.from_pretrained_url(
            cfg.model,
            pretrained_path,
            map_location=torch.device("cpu"),
        )
    model.eval()
    model.to(cfg.device)

    dataset = ProteinDomainDataset(
        root=cfg.datadir,
        cath_data=cath_data
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    protein_repr_all, labels_cath, superfamilies = compute_repr(data_loader, model, cfg)

    # Save embeddings and labels
    save_path = Path("./data/embeds")
    save_path.mkdir(parents=True, exist_ok=True)

    embeddings_file = save_path / "protein_representations.pt"
    labels_file = save_path / "labels_cath.npy"
    superfam_file = save_path / "superfamilies.npy"

    torch.save(protein_repr_all, embeddings_file)
    np.save(labels_file, labels_cath)
    np.save(superfam_file, superfamilies)

    print(f"Embeddings saved to {embeddings_file}")
    print(f"Labels saved to {labels_file}")


if __name__ == "__main__":
    main()