#!/usr/bin/env python3
"""Create a reduced copy of a cached RelBench rel-amazon dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/cache/relbench/rel-amazon"),
        help="Source rel-amazon cache directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/relbench/rel-amazon_since_2017-01-28"),
        help="Output directory for the filtered dataset.",
    )
    parser.add_argument("--task", type=str, default="user-item-rate")
    parser.add_argument("--cutoff", type=str, default="2017-01-28")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument(
        "--task_keep_ratio",
        type=float,
        default=0.2,
        help="Fraction of rows to keep in each train/val/test task split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_archives",
        action="store_true",
        help="Copy original db.zip and task zip files into the output.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    (path / "db").mkdir(parents=True)


def writer_for(source_path: Path, output_path: Path) -> pq.ParquetWriter:
    schema = pq.ParquetFile(source_path).schema_arrow
    return pq.ParquetWriter(output_path, schema=schema, compression="snappy")


def add_int_values(values: pa.Array | pa.ChunkedArray, target: set[int]) -> None:
    if isinstance(values, pa.ChunkedArray):
        chunks = values.chunks
    else:
        chunks = [values]
    for chunk in chunks:
        chunk = pc.drop_null(chunk)
        if len(chunk) == 0:
            continue
        target.update(int(value) for value in chunk.to_pylist())


def add_list_int_values(values: pa.Array | pa.ChunkedArray,
                        target: set[int]) -> None:
    flattened = pc.list_flatten(values)
    add_int_values(flattened, target)


def add_task_pairs(batch: pa.RecordBatch, pair_keys: set[tuple[int, int]],
                   customer_ids: set[int], product_ids: set[int]) -> None:
    customers = batch.column("customer_id").to_pylist()
    product_lists = batch.column("product_id").to_pylist()
    for customer_id, products in zip(customers, product_lists):
        if customer_id is None or products is None:
            continue
        customer_id = int(customer_id)
        customer_ids.add(customer_id)
        for product_id in products:
            if product_id is None:
                continue
            product_id = int(product_id)
            product_ids.add(product_id)
            pair_keys.add((customer_id, product_id))


def sampled_indices(total_rows: int, keep_ratio: float,
                    seed: int) -> np.ndarray:
    if not 0 < keep_ratio <= 1:
        raise ValueError("--task_keep_ratio must be in the range (0, 1]")
    keep_rows = int(round(total_rows * keep_ratio))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=keep_rows, replace=False))


def collect_sampled_task_entities(
    source_path: Path,
    batch_size: int,
    kept_indices: np.ndarray,
    customer_ids: set[int],
    product_ids: set[int],
    pair_keys: set[tuple[int, int]],
) -> int:
    parquet_file = pq.ParquetFile(source_path)
    rows = 0
    offset = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        use_threads=False,
    ):
        batch_indices = np.arange(offset, offset + batch.num_rows)
        mask = np.isin(batch_indices, kept_indices, assume_unique=True)
        offset += batch.num_rows
        filtered = batch.filter(pa.array(mask))
        if filtered.num_rows == 0:
            continue
        rows += filtered.num_rows
        add_task_pairs(filtered, pair_keys, customer_ids, product_ids)
    return rows


def replace_column(table: pa.Table, name: str,
                   values: pa.Array | list) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    array = values if isinstance(values, pa.Array) else pa.array(
        values, type=field.type)
    return table.set_column(index, field, array)


def write_filtered_by_ids_remapped(
    source_path: Path,
    output_path: Path,
    id_col: str,
    ids: set[int],
    batch_size: int,
) -> tuple[dict[int, int], int]:
    schema = pq.ParquetFile(source_path).schema_arrow
    id_type = schema.field(id_col).type
    sorted_ids = sorted(ids)
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
    value_set = pa.array(sorted_ids, type=id_type)
    parquet_file = pq.ParquetFile(source_path)
    rows = 0
    with writer_for(source_path, output_path) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            use_threads=False,
        ):
            mask = pc.is_in(batch.column(id_col), value_set=value_set)
            filtered = batch.filter(mask)
            if filtered.num_rows == 0:
                continue
            table = pa.Table.from_batches([filtered])
            remapped_ids = [
                id_map[int(value)] for value in table.column(id_col).to_pylist()
            ]
            table = replace_column(table, id_col, remapped_ids)
            writer.write_table(table)
            rows += filtered.num_rows
    return id_map, rows


def write_sampled_task_split_remapped(
    source_path: Path,
    output_path: Path,
    batch_size: int,
    kept_indices: np.ndarray,
    customer_id_map: dict[int, int],
    product_id_map: dict[int, int],
) -> int:
    parquet_file = pq.ParquetFile(source_path)
    rows = 0
    offset = 0
    with writer_for(source_path, output_path) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            use_threads=False,
        ):
            batch_indices = np.arange(offset, offset + batch.num_rows)
            mask = np.isin(batch_indices, kept_indices, assume_unique=True)
            offset += batch.num_rows
            filtered = batch.filter(pa.array(mask))
            if filtered.num_rows == 0:
                continue
            table = pa.Table.from_batches([filtered])
            remapped_customers = [
                customer_id_map[int(value)]
                for value in table.column("customer_id").to_pylist()
            ]
            remapped_products = [
                [product_id_map[int(product_id)] for product_id in products]
                for products in table.column("product_id").to_pylist()
            ]
            table = replace_column(table, "customer_id", remapped_customers)
            table = replace_column(table, "product_id", remapped_products)
            writer.write_table(table)
            rows += filtered.num_rows
    return rows


def write_review_for_task_pairs_remapped(
    source_path: Path,
    output_path: Path,
    batch_size: int,
    pair_keys: set[tuple[int, int]],
    customer_id_map: dict[int, int],
    product_id_map: dict[int, int],
) -> int:
    parquet_file = pq.ParquetFile(source_path)
    rows = 0
    with writer_for(source_path, output_path) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            use_threads=False,
        ):
            customers = batch.column("customer_id").to_pylist()
            products = batch.column("product_id").to_pylist()
            mask = [
                (customer_id is not None and product_id is not None and
                 (int(customer_id), int(product_id)) in pair_keys)
                for customer_id, product_id in zip(customers, products)
            ]
            filtered = batch.filter(pa.array(mask))
            if filtered.num_rows == 0:
                continue
            table = pa.Table.from_batches([filtered])
            remapped_customers = [
                customer_id_map[int(value)]
                for value in table.column("customer_id").to_pylist()
            ]
            remapped_products = [
                product_id_map[int(value)]
                for value in table.column("product_id").to_pylist()
            ]
            table = replace_column(table, "customer_id", remapped_customers)
            table = replace_column(table, "product_id", remapped_products)
            writer.write_table(table)
            rows += filtered.num_rows
    return rows


def copy_zip_if_exists(source: Path, output: Path, name: str) -> None:
    source_path = source / name
    if source_path.exists():
        shutil.copy2(source_path, output / name)


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    prepare_output(args.output, args.overwrite)

    customer_ids: set[int] = set()
    product_ids: set[int] = set()
    pair_keys: set[tuple[int, int]] = set()

    task_source = args.source / "tasks" / args.task
    task_output = args.output / "tasks" / args.task
    task_output.mkdir(parents=True)

    task_indices: dict[str, np.ndarray] = {}
    task_rows: dict[str, int] = {}
    for split_idx, split in enumerate(["train", "val", "test"]):
        split_path = task_source / f"{split}.parquet"
        total_rows = pq.ParquetFile(split_path).metadata.num_rows
        kept_indices = sampled_indices(
            total_rows=total_rows,
            keep_ratio=args.task_keep_ratio,
            seed=args.seed + split_idx,
        )
        task_indices[split] = kept_indices
        rows = collect_sampled_task_entities(
            source_path=task_source / f"{split}.parquet",
            batch_size=args.batch_size,
            kept_indices=kept_indices,
            customer_ids=customer_ids,
            product_ids=product_ids,
            pair_keys=pair_keys,
        )
        task_rows[split] = rows

    customer_id_map, customer_rows = write_filtered_by_ids_remapped(
        source_path=args.source / "db" / "customer.parquet",
        output_path=args.output / "db" / "customer.parquet",
        id_col="customer_id",
        ids=customer_ids,
        batch_size=args.batch_size,
    )
    product_id_map, product_rows = write_filtered_by_ids_remapped(
        source_path=args.source / "db" / "product.parquet",
        output_path=args.output / "db" / "product.parquet",
        id_col="product_id",
        ids=product_ids,
        batch_size=args.batch_size,
    )

    remapped_task_rows: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        rows = write_sampled_task_split_remapped(
            source_path=task_source / f"{split}.parquet",
            output_path=task_output / f"{split}.parquet",
            batch_size=args.batch_size,
            kept_indices=task_indices[split],
            customer_id_map=customer_id_map,
            product_id_map=product_id_map,
        )
        remapped_task_rows[split] = rows

    review_rows = write_review_for_task_pairs_remapped(
        source_path=args.source / "db" / "review.parquet",
        output_path=args.output / "db" / "review.parquet",
        batch_size=args.batch_size,
        pair_keys=pair_keys,
        customer_id_map=customer_id_map,
        product_id_map=product_id_map,
    )

    if args.copy_archives:
        copy_zip_if_exists(args.source, args.output, "db.zip")
        copy_zip_if_exists(args.source / "tasks", args.output / "tasks",
                           f"{args.task}.zip")

    summary_path = args.output / "subset_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"source: {args.source}\n")
        f.write(f"output: {args.output}\n")
        f.write(f"cutoff: {cutoff} (not applied to task sampling)\n")
        f.write(f"task_keep_ratio: {args.task_keep_ratio}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"review_rows: {review_rows:,}\n")
        f.write(f"customer_rows: {customer_rows:,}\n")
        f.write(f"product_rows: {product_rows:,}\n")
        for split, rows in remapped_task_rows.items():
            f.write(f"{split}_rows: {rows:,}\n")
        f.write(f"unique_task_pairs: {len(pair_keys):,}\n")
        f.write(f"unique_customer_ids: {len(customer_ids):,}\n")
        f.write(f"unique_product_ids: {len(product_ids):,}\n")

    print(f"Wrote filtered dataset to {args.output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
