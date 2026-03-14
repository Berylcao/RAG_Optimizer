"""
CLI entry point for downloading local embedding and reranker models.
"""
import argparse
import logging

from src.utils.hf_model_downloader import download_all_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download local embedding and reranker models.")
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Use the hf-mirror endpoint for downloads.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Download only the specified model labels, e.g. --only bge-small bge-reranker-base",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN env var when omitted.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()
    download_all_models(
        use_mirror=args.use_mirror,
        only_labels=args.only,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
