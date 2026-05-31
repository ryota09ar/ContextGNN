"""Train and evaluate ContextGNN on cached Yelp transductive data."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch_frame
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from tqdm import tqdm

from contextgnn.nn.models import ContextGNN
from contextgnn.utils import RHSEmbeddingMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cache/yelp/transductive_recent_2020-02-01",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_epochs_interval", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--rhs_sample_size", type=int, default=1000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_train_items", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    return parser.parse_args()


def make_embedding_tf(feat: Tensor, col_name: str) -> TensorFrame:
    num_rows, dim = feat.size()
    emb = MultiEmbeddingTensor(
        num_rows=num_rows,
        num_cols=1,
        values=feat,
        offset=torch.tensor([0, dim]),
    )
    return TensorFrame(
        feat_dict={torch_frame.embedding: emb},
        col_names_dict={torch_frame.embedding: [col_name]},
        num_rows=num_rows,
    )


def make_positive_index(
    user_ids: Tensor,
    item_ids: Tensor,
    num_users: int,
) -> List[Tensor]:
    items_by_user: List[List[int]] = [[] for _ in range(num_users)]
    for user, item in zip(user_ids.tolist(), item_ids.tolist()):
        items_by_user[user].append(item)
    return [torch.tensor(items, dtype=torch.long) for items in items_by_user]


def clear_rhs_cache(model: ContextGNN) -> None:
    model.rhs_embedding._cached_rhs_embedding = None


def attach_contextgnn_batch_fields(
    batch: HeteroData,
    train_positives: List[Tensor],
) -> HeteroData:
    batch_size = batch["user"].batch_size
    device = batch["user"].n_id.device
    batch["user"].seed_time = torch.zeros(batch_size, device=device)

    user_batch = torch.zeros(batch["user"].num_nodes, dtype=torch.long)
    user_batch[:batch_size] = torch.arange(batch_size)
    batch["user"].batch = user_batch.to(device)

    item_to_batch: Dict[int, int] = {}
    seed_user_ids = batch["user"].n_id[:batch_size].cpu()
    for batch_idx, user_id in enumerate(seed_user_ids.tolist()):
        for item_id in train_positives[user_id].tolist():
            item_to_batch.setdefault(item_id, batch_idx)

    item_batch = torch.zeros(batch["item"].num_nodes, dtype=torch.long)
    for local_item_idx, item_id in enumerate(batch["item"].n_id.cpu().tolist()):
        item_batch[local_item_idx] = item_to_batch.get(item_id, 0)
    batch["item"].batch = item_batch.to(device)
    return batch


def batch_positives(
    input_user_ids: Tensor,
    positives: List[Tensor],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    src_batches = []
    dst_indices = []
    for batch_idx, user_id in enumerate(input_user_ids.tolist()):
        item_ids = positives[user_id]
        if item_ids.numel() == 0:
            continue
        src_batches.append(
            torch.full((item_ids.numel(),), batch_idx, dtype=torch.long))
        dst_indices.append(item_ids)
    if not src_batches:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return (
        torch.cat(src_batches).to(device),
        torch.cat(dst_indices).to(device),
    )


def build_data(metadata: dict) -> Tuple[HeteroData, Dict[str, dict]]:
    item_feat = metadata["item_feature_table"].float()
    train_user = metadata["train_edges"]["user_ids"].long()
    train_item = metadata["train_edges"]["item_ids"].long()
    num_users = len(metadata["user_index"])
    num_items = len(metadata["business_index"])

    user_feat = torch.zeros(num_users, item_feat.size(1), dtype=item_feat.dtype)
    user_feat.index_add_(0, train_user, item_feat[train_item])
    counts = torch.bincount(train_user, minlength=num_users).clamp_min(1)
    user_feat = user_feat / counts.view(-1, 1)

    data = HeteroData()
    data["user"].num_nodes = num_users
    data["item"].num_nodes = num_items
    data["user"].time = torch.zeros(num_users)
    data["item"].time = torch.zeros(num_items)
    data["user"].tf = make_embedding_tf(user_feat, "user_history_embedding")
    data["item"].tf = make_embedding_tf(item_feat, "item_embedding")
    data["user", "rates", "item"].edge_index = torch.stack(
        [train_user, train_item], dim=0)
    data["item", "rev_rates", "user"].edge_index = torch.stack(
        [train_item, train_user], dim=0)

    dim = int(metadata["item_feature_dim"])
    col_stats_dict = {
        "user": {
            "user_history_embedding": {
                StatType.EMB_DIM: dim,
            },
        },
        "item": {
            "item_embedding": {
                StatType.EMB_DIM: dim,
            },
        },
    }
    return data, col_stats_dict


def make_loader(
    data: HeteroData,
    user_ids: Tensor,
    num_neighbors: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> NeighborLoader:
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=("user", user_ids.unique(sorted=True)),
        subgraph_type="bidirectional",
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: ContextGNN,
    loader: NeighborLoader,
    positives: List[Tensor],
    train_positives: List[Tensor],
    eval_k: int,
    device: torch.device,
    filter_train_items: bool,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    clear_rhs_cache(model)
    recall_sum = ndcg_sum = map_sum = hit_sum = 0.0
    num_eval_users = 0

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        batch = attach_contextgnn_batch_fields(batch, train_positives)
        user_ids = batch["user"].n_id[:batch["user"].batch_size].cpu()
        scores = model(batch, "user", "item").detach()

        if filter_train_items:
            for row, user_id in enumerate(user_ids.tolist()):
                seen = train_positives[user_id]
                if seen.numel() > 0:
                    scores[row, seen.to(device)] = -float("inf")

        _, pred = torch.topk(scores, k=eval_k, dim=1)
        pred = pred.cpu()

        for row, user_id in enumerate(user_ids.tolist()):
            target = positives[user_id]
            if target.numel() == 0:
                continue
            target_set = set(target.tolist())
            hits = [1 if item in target_set else 0 for item in pred[row].tolist()]
            num_hits = sum(hits)
            if num_hits == 0:
                num_eval_users += 1
                continue

            recall_sum += num_hits / min(len(target_set), eval_k)
            hit_sum += 1.0

            dcg = sum(hit / math.log2(rank + 2)
                      for rank, hit in enumerate(hits))
            ideal_hits = min(len(target_set), eval_k)
            idcg = sum(1.0 / math.log2(rank + 2)
                       for rank in range(ideal_hits))
            ndcg_sum += dcg / idcg

            precision_sum = 0.0
            running_hits = 0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    running_hits += 1
                    precision_sum += running_hits / rank
            map_sum += precision_sum / min(len(target_set), eval_k)
            num_eval_users += 1

    if num_eval_users == 0:
        return {
            f"hit@{eval_k}": 0.0,
            f"recall@{eval_k}": 0.0,
            f"ndcg@{eval_k}": 0.0,
            f"map@{eval_k}": 0.0,
        }
    return {
        f"hit@{eval_k}": hit_sum / num_eval_users,
        f"recall@{eval_k}": recall_sum / num_eval_users,
        f"ndcg@{eval_k}": ndcg_sum / num_eval_users,
        f"map@{eval_k}": map_sum / num_eval_users,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    metadata = torch.load(data_dir / "metadata.pt", map_location="cpu")
    split_edges = {
        split: torch.load(data_dir / f"{split}.pt", map_location="cpu")
        for split in ["train", "valid", "test"]
    }

    data, col_stats_dict = build_data(metadata)
    num_neighbors = [
        int(args.num_neighbors // 2**i) for i in range(args.num_layers)
    ]
    loaders = {
        split: make_loader(
            data=data,
            user_ids=edges["user_ids"].long(),
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
        )
        for split, edges in split_edges.items()
    }

    positives = {
        split: make_positive_index(
            edges["user_ids"].long(),
            edges["item_ids"].long(),
            data["user"].num_nodes,
        )
        for split, edges in split_edges.items()
    }

    model = ContextGNN(
        data=data,
        col_stats_dict=col_stats_dict,
        rhs_emb_mode=RHSEmbeddingMode.FUSION,
        dst_entity_table="item",
        num_nodes=data["item"].num_nodes,
        num_layers=args.num_layers,
        channels=args.channels,
        aggr=args.aggr,
        norm="layer_norm",
        embedding_dim=64,
        torch_frame_model_kwargs={
            "channels": 128,
            "num_layers": 4,
        },
        rhs_sample_size=args.rhs_sample_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state_dict = None
    best_valid_metric = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_count = steps = 0
        total_steps = min(len(loaders["train"]), args.max_steps_per_epoch)

        for batch in tqdm(loaders["train"], total=total_steps, desc="Train"):
            batch = batch.to(device)
            batch = attach_contextgnn_batch_fields(batch, positives["train"])
            input_user_ids = batch["user"].n_id[:batch["user"].batch_size].cpu()
            src_batch, dst_index = batch_positives(
                input_user_ids,
                positives["train"],
                device,
            )
            if dst_index.numel() == 0:
                continue

            optimizer.zero_grad()
            logits, lhs_y_batch, rhs_y_index = model.forward_sample_softmax(
                batch, "user", "item", src_batch, dst_index)
            edge_label_index = torch.stack([lhs_y_batch, rhs_y_index], dim=0)
            loss = sparse_cross_entropy(logits, edge_label_index)
            loss.backward()
            optimizer.step()

            count = int(batch["user"].batch_size)
            total_loss += float(loss.detach()) * count
            total_count += count
            steps += 1
            if steps >= args.max_steps_per_epoch:
                break

        train_loss = total_loss / total_count if total_count else float("nan")
        if epoch % args.eval_epochs_interval == 0:
            valid_metrics = evaluate(
                model=model,
                loader=loaders["valid"],
                positives=positives["valid"],
                train_positives=positives["train"],
                eval_k=args.eval_k,
                device=device,
                filter_train_items=args.filter_train_items,
                desc="Valid",
            )
            print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, "
                  f"Valid metrics: {valid_metrics}")

            valid_metric = valid_metrics[f"map@{args.eval_k}"]
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                best_state_dict = {
                    key: value.cpu()
                    for key, value in model.state_dict().items()
                }

    if best_state_dict is None:
        raise RuntimeError("No validation checkpoint was selected.")

    model.load_state_dict(best_state_dict)
    clear_rhs_cache(model)
    model = model.to(device)
    valid_metrics = evaluate(
        model=model,
        loader=loaders["valid"],
        positives=positives["valid"],
        train_positives=positives["train"],
        eval_k=args.eval_k,
        device=device,
        filter_train_items=args.filter_train_items,
        desc="Best valid",
    )
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        positives=positives["test"],
        train_positives=positives["train"],
        eval_k=args.eval_k,
        device=device,
        filter_train_items=args.filter_train_items,
        desc="Test",
    )
    print(f"Best valid metrics: {valid_metrics}")
    print(f"Best test metrics: {test_metrics}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state_dict, save_dir / "contextgnn_yelp_best.pt")
        with open(save_dir / "metrics.json", "w") as f:
            json.dump({
                "best_valid_metrics": valid_metrics,
                "best_test_metrics": test_metrics,
                "args": vars(args),
            }, f, indent=2)


if __name__ == "__main__":
    main()
