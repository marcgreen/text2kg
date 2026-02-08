#!/usr/bin/env python3
"""
Text-to-Knowledge-Graph pipeline using Claude Code CLI.

Usage:
    python run_pipeline.py                          # uses cached data from data_output/
    python run_pipeline.py --regenerate             # re-extracts graph via Claude CLI
    python run_pipeline.py --regenerate --data-dir mydata
    python run_pipeline.py --regenerate --model sonnet
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

# Ensure project root is on the path so helpers/ and claude_client can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.df_helpers import documents2Dataframe, df2Graph, graph2Df


def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2


def main():
    parser = argparse.ArgumentParser(description="Text to Knowledge Graph pipeline")
    parser.add_argument(
        "--data-dir",
        default="cureus",
        help="Subdirectory under data_input/ containing source documents (default: cureus)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Subdirectory under data_output/ for results (defaults to --data-dir value)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-extract the graph using Claude CLI (otherwise reads cached CSV)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to pass to Claude CLI (e.g. 'sonnet', 'opus'). None = CLI default.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Text chunk size in characters (default: 1500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap between chunks (default: 150)",
    )
    parser.add_argument(
        "--output-html",
        default="./docs/index.html",
        help="Path for the output HTML graph (default: ./docs/index.html)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    inputdirectory = Path(f"./data_input/{args.data_dir}")
    outputdirectory = Path(f"./data_output/{out_dir}")

    if not inputdirectory.exists():
        print(f"Error: Input directory {inputdirectory} does not exist.")
        sys.exit(1)

    # ---- Step 1: Load and chunk documents ----
    print(f"Loading documents from {inputdirectory} ...")
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader(
        str(inputdirectory),
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True,
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    pages = splitter.split_documents(documents)
    print(f"Number of chunks = {len(pages)}")

    df = documents2Dataframe(pages)
    print(f"Chunks dataframe shape: {df.shape}")

    # ---- Step 2: Extract graph (or load cached) ----
    if args.regenerate:
        print(f"\nExtracting knowledge graph via Claude CLI (model={args.model}) ...")
        print(f"Processing {len(df)} chunks - this will make one CLI call per chunk.\n")
        concepts_list = df2Graph(df, model=args.model)
        dfg1 = graph2Df(concepts_list)

        os.makedirs(outputdirectory, exist_ok=True)
        dfg1.to_csv(outputdirectory / "graph.csv", sep="|", index=False)
        df.to_csv(outputdirectory / "chunks.csv", sep="|", index=False)
        print(f"\nSaved graph.csv and chunks.csv to {outputdirectory}")
    else:
        csv_path = outputdirectory / "graph.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run with --regenerate first.")
            sys.exit(1)
        print(f"Loading cached graph from {csv_path} ...")
        dfg1 = pd.read_csv(csv_path, sep="|")

    dfg1.replace("", np.nan, inplace=True)
    dfg1.dropna(subset=["node_1", "node_2", "edge"], inplace=True)
    dfg1["count"] = 4
    print(f"Graph edges (LLM-extracted): {dfg1.shape[0]}")

    # ---- Step 3: Contextual proximity ----
    print("Calculating contextual proximity ...")
    dfg2 = contextual_proximity(dfg1)
    print(f"Contextual proximity edges: {dfg2.shape[0]}")

    # ---- Step 4: Merge edges ----
    dfg = pd.concat([dfg1, dfg2], axis=0)
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ",".join, "count": "sum"})
        .reset_index()
    )
    print(f"Total merged edges: {dfg.shape[0]}")

    # ---- Step 5: Build NetworkX graph ----
    nodes = pd.concat([dfg["node_1"], dfg["node_2"]], axis=0).unique()
    print(f"Total unique nodes: {len(nodes)}")

    G = nx.Graph()
    for node in nodes:
        G.add_node(str(node))
    for _, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row["count"] / 4,
        )

    # ---- Step 6: Community detection ----
    print("Detecting communities ...")
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print(f"Number of communities: {len(communities)}")

    # ---- Step 7: Assign colors ----
    import seaborn as sns

    palette = sns.color_palette("hls", len(communities)).as_hex()
    random.shuffle(palette)
    for community in communities:
        color = palette.pop()
        for node in community:
            G.nodes[node]["color"] = color
            G.nodes[node]["size"] = G.degree[node]

    # ---- Step 8: Visualize ----
    from pyvis.network import Network

    os.makedirs(os.path.dirname(args.output_html) or ".", exist_ok=True)

    net = Network(
        notebook=False,
        cdn_resources="remote",
        height="900px",
        width="100%",
        select_menu=True,
        filter_menu=False,
    )
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)
    net.show_buttons(filter_=["physics"])
    net.show(args.output_html, notebook=False)

    print(f"\nDone! Knowledge graph written to {args.output_html}")
    print(f"Open it in a browser to explore the interactive visualization.")


if __name__ == "__main__":
    main()
