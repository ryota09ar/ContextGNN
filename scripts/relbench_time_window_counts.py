#!/usr/bin/env python3
"""Count RelBench rows retained by rolling time windows.

The script reads only timestamp columns from parquet files, so it can inspect
large RelBench datasets without materializing text columns in memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class TimeTable:
    name: str
    path: Path
    time_col: str
    rows: int
    min_time: pd.Timestamp
    max_time: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/cache/relbench/rel-amazon"),
        help="RelBench dataset cache directory containing db/*.parquet.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="user-item-rate",
        help="Task directory under tasks/. Used only with --include_tasks.",
    )
    parser.add_argument(
        "--include_tasks",
        action="store_true",
        help="Also count train/val/test task parquet files.",
    )
    parser.add_argument(
        "--months_step",
        type=int,
        default=2,
        help="Window step in months.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/relbench_time_window_counts.txt"),
        help="Text file to write the report.",
    )
    return parser.parse_args()


def parquet_files(dataset_dir: Path, task: str, include_tasks: bool) -> list[Path]:
    paths = sorted((dataset_dir / "db").glob("*.parquet"))
    if include_tasks:
        paths.extend(sorted((dataset_dir / "tasks" / task).glob("*.parquet")))
    return paths


def get_time_col(parquet_file: pq.ParquetFile) -> str | None:
    metadata = parquet_file.schema_arrow.metadata or {}
    raw_time_col = metadata.get(b"time_col")
    if raw_time_col is not None:
        time_col = json.loads(raw_time_col.decode("utf-8"))
        if time_col is not None:
            return str(time_col)

    for field in parquet_file.schema_arrow:
        if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
            return field.name
    return None


def iter_time_arrays(path: Path, time_col: str) -> Iterable[pa.ChunkedArray]:
    parquet_file = pq.ParquetFile(path)
    for row_group_idx in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group_idx, columns=[time_col])
        yield table[time_col]


def min_max_time(path: Path, time_col: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    for array in iter_time_arrays(path, time_col):
        stats = pc.min_max(array).as_py()
        if stats["min"] is None or stats["max"] is None:
            continue
        chunk_min = pd.Timestamp(stats["min"])
        chunk_max = pd.Timestamp(stats["max"])
        min_time = chunk_min if min_time is None else min(min_time, chunk_min)
        max_time = chunk_max if max_time is None else max(max_time, chunk_max)

    if min_time is None or max_time is None:
        raise ValueError(f"No non-null timestamp values found in {path}:{time_col}")
    return min_time, max_time


def discover_time_tables(paths: list[Path]) -> list[TimeTable]:
    tables: list[TimeTable] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        time_col = get_time_col(parquet_file)
        if time_col is None:
            continue
        min_time, max_time = min_max_time(path, time_col)
        tables.append(
            TimeTable(
                name=path.stem,
                path=path,
                time_col=time_col,
                rows=parquet_file.metadata.num_rows,
                min_time=min_time,
                max_time=max_time,
            )
        )
    return tables


def read_sorted_timestamps(path: Path, time_col: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for array in iter_time_arrays(path, time_col):
        for chunk in array.chunks:
            chunk = pc.drop_null(chunk)
            if len(chunk) == 0:
                continue
            arrays.append(chunk.to_numpy(zero_copy_only=False))
    if not arrays:
        return np.array([], dtype="datetime64[ns]")
    values = np.concatenate(arrays).astype("datetime64[ns]", copy=False)
    values.sort()
    return values


def count_rows_since_many(path: Path, time_col: str,
                          cutoffs: list[pd.Timestamp]) -> list[int]:
    values = read_sorted_timestamps(path, time_col)
    total = len(values)
    counts: list[int] = []
    for cutoff in cutoffs:
        cutoff_value = np.datetime64(cutoff.to_datetime64())
        first_retained_idx = int(np.searchsorted(values, cutoff_value,
                                                 side="left"))
        counts.append(total - first_retained_idx)
    return counts


def build_cutoffs(max_time: pd.Timestamp, min_time: pd.Timestamp,
                  months_step: int) -> list[pd.Timestamp]:
    if months_step <= 0:
        raise ValueError("--months_step must be positive")

    cutoffs: list[pd.Timestamp] = []
    cutoff = max_time
    while cutoff >= min_time:
        cutoffs.append(cutoff)
        cutoff = cutoff - pd.DateOffset(months=months_step)
    if cutoffs[-1] != min_time:
        cutoffs.append(min_time)
    return cutoffs


def format_ratio(count: int, total: int) -> str:
    pct = 100.0 * count / total if total else 0.0
    return f"{count:,}/{total:,} ({pct:.2f}%)"


def main() -> None:
    args = parse_args()
    paths = parquet_files(args.dataset_dir, args.task, args.include_tasks)
    tables = discover_time_tables(paths)
    if not tables:
        raise ValueError(f"No time-aware parquet tables found under {args.dataset_dir}")

    global_min_time = min(table.min_time for table in tables)
    global_max_time = max(table.max_time for table in tables)
    total_rows = sum(table.rows for table in tables)
    cutoffs = build_cutoffs(global_max_time, global_min_time, args.months_step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("RelBench time-window row counts\n")
        f.write(f"dataset_dir: {args.dataset_dir}\n")
        f.write(f"include_tasks: {args.include_tasks}\n")
        f.write(f"months_step: {args.months_step}\n")
        f.write(f"global_min_time: {global_min_time}\n")
        f.write(f"global_max_time: {global_max_time}\n")
        f.write(f"time-aware_total_rows: {total_rows:,}\n\n")

        f.write("Time-aware tables\n")
        f.write("table\ttime_col\trows\tmin_time\tmax_time\tpath\n")
        for table in tables:
            f.write(
                f"{table.name}\t{table.time_col}\t{table.rows:,}\t"
                f"{table.min_time}\t{table.max_time}\t{table.path}\n"
            )
        f.write("\n")

        header = ["cutoff", "total_retained/total"]
        header.extend(f"{table.name}_retained/{table.name}_total"
                      for table in tables)
        f.write("\t".join(header) + "\n")

        counts_by_table = [
            count_rows_since_many(table.path, table.time_col, cutoffs)
            for table in tables
        ]
        for cutoff_idx, cutoff in enumerate(cutoffs):
            table_counts = [counts[cutoff_idx] for counts in counts_by_table]
            retained_total = sum(table_counts)
            row = [str(cutoff), format_ratio(retained_total, total_rows)]
            row.extend(
                format_ratio(count, table.rows)
                for count, table in zip(table_counts, tables)
            )
            f.write("\t".join(row) + "\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
