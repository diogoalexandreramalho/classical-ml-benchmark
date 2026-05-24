import argparse

from data_science import pipeline, unsupervised
from data_science.datasets import DATASETS

TASKS = ("classification", "unsupervised")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the classification or unsupervised pipeline on a dataset.",
    )
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()

    dataframe = DATASETS[args.dataset].read()
    if args.task == "classification":
        pipeline.classification(dataframe, args.dataset)
    elif args.task == "unsupervised":
        unsupervised.run(args.dataset, dataframe)


if __name__ == "__main__":
    main()
