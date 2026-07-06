import argparse
import sys
from typing import List, Union

from huggingface_hub import scan_cache_dir


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


def ask_for_confirmation(message: str) -> bool:
    y = ("y", "yes", "1")
    n = ("n", "no", "0", "")
    full_message = f"{message} (y/n) "
    while True:
        answer = input(full_message).lower()
        if answer in y:
            return True
        if answer in n:
            return False
        print(f"Invalid input. Must be one of: yes/no/y/n or empty for no")


def _save_draft_heads_cmd(argv):
    import mlx.core as mx
    import mlx.nn as nn

    from .utils import (
        _download,
        _load_saved_draft_lm_heads,
        _maybe_build_mtp_draft_lm_heads,
        _maybe_quantize_mtp_fc,
        load_model,
    )

    parser = argparse.ArgumentParser(
        prog="mlx_lm.manage save-draft-heads",
        description="Pre-build and save MTP draft lm_heads to draft_heads_{N}bit.safetensors.",
    )
    parser.add_argument("--model", required=True, help="Model path or HuggingFace repo ID.")
    parser.add_argument(
        "--bits",
        required=True,
        help="Comma-separated bit widths to build, e.g. '4' or '4,8'.",
    )
    parser.add_argument(
        "--mtp-fc-bits",
        type=int,
        default=-1,
        help="Quantize MTP fc layer to this precision before building heads.",
    )
    args = parser.parse_args(argv)

    bits_list = [int(b.strip()) for b in args.bits.split(",")]
    model_path = _download(args.model)

    print(f"Loading model from {model_path} ...")
    model, config = load_model(model_path, lazy=False)
    _maybe_quantize_mtp_fc(model, config, bits=args.mtp_fc_bits)
    _load_saved_draft_lm_heads(model, model_path)

    lm = getattr(model, "language_model", model)
    if not hasattr(lm, "mtp"):
        print("Model has no MTP module. Nothing to save.")
        return

    for bits in bits_list:
        _maybe_build_mtp_draft_lm_heads(model, [bits])
        heads_dict = getattr(lm, "_mtp_draft_lm_heads", {})
        head = heads_dict.get(bits)
        if head is None:
            print(f"  {bits}-bit: skipped (matches native precision).")
            continue
        out_path = model_path / f"draft_heads_{bits}bit.safetensors"
        mx.eval(head.weight, head.scales, head.biases)
        mx.save_safetensors(
            str(out_path),
            {"weight": head.weight, "scales": head.scales, "biases": head.biases},
        )
        print(f"  {bits}-bit: saved → {out_path}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "save-draft-heads":
        _save_draft_heads_cmd(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="MLX Model Cache.")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan Hugging Face cache for mlx models.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete models matching the given pattern.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Model repos contain the pattern.",
        default="mlx",
    )

    args = parser.parse_args()

    if args.scan:
        print(f'Scanning Hugging Face cache for models with pattern "{args.pattern}".')
        hf_cache_info = scan_cache_dir()
        print(
            tabulate(
                rows=[
                    [
                        repo.repo_id,
                        repo.repo_type,
                        "{:>12}".format(repo.size_on_disk_str),
                        repo.nb_files,
                        repo.last_accessed_str,
                        repo.last_modified_str,
                        str(repo.repo_path),
                    ]
                    for repo in sorted(
                        hf_cache_info.repos, key=lambda repo: repo.repo_path
                    )
                    if args.pattern in repo.repo_id
                ],
                headers=[
                    "REPO ID",
                    "REPO TYPE",
                    "SIZE ON DISK",
                    "NB FILES",
                    "LAST_ACCESSED",
                    "LAST_MODIFIED",
                    "LOCAL PATH",
                ],
            )
        )

    if args.delete:
        print(f'Deleting models matching pattern "{args.pattern}"')
        hf_cache_info = scan_cache_dir()

        repos = [
            repo
            for repo in sorted(hf_cache_info.repos, key=lambda repo: repo.repo_path)
            if args.pattern in repo.repo_id
        ]
        if repos:
            print("\nFound the following models:")
            print(
                tabulate(
                    rows=[
                        [
                            repo.repo_id,
                            repo.size_on_disk_str,  # Added size information
                            str(repo.repo_path),
                        ]
                        for repo in repos
                    ],
                    headers=[
                        "REPO ID",
                        "SIZE",  # Added size header
                        "LOCAL PATH",
                    ],
                )
            )

            confirmed = ask_for_confirmation(
                "\nAre you sure you want to delete these models?"
            )
            if confirmed:
                for model_info in repos:
                    print(f"\nDeleting {model_info.repo_id}...")
                    for revision in sorted(
                        model_info.revisions, key=lambda revision: revision.commit_hash
                    ):
                        strategy = hf_cache_info.delete_revisions(revision.commit_hash)
                        strategy.execute()
                print("\nModel(s) deleted successfully.")
            else:
                print("\nDeletion cancelled - no changes made.")
        else:
            print(f'No models found matching pattern "{args.pattern}"')


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.manage...` directly is deprecated."
        " Use `mlx_lm.manage...` or `python -m mlx_lm manage ...` instead."
    )
    main()
