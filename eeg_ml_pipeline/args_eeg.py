import argparse


def args():
    parser = argparse.ArgumentParser(
        description="Perform Feature Extraction and Feature Selection on input dataset"
    )

    # Defining the model
    parser.add_argument(
        "--dataset",
        default="DEAP", 
        type=str, 
        help="Dataset to be analysed: DEAP, DREAMER or OASIS?"
    )

    parser.add_argument(
        "--window",
        default=4, 
        type=int, 
        help="window length"
    )
    parser.add_argument(
        "--stride",
        default=2,
        type=int,
        help="stride",
    )
    parser.add_argument(
        "--sfreq",
        default=0.35,
        type=int,
        help="Sampling Frequency",
    )
    parser.add_argument(
        "--model",
        default='svc',
        type=str,
        help="Enter sklearn model: svc or rfr?",
    )
    parser.add_argument(
        "--label",
        default=0,
        type=int,
        help="Y-Label",
    )

    parser.add_argument(
        "--approach",
        default="byclassifier",
        type=str,
        help="Approach: byclassifier or byfs?",
    )

    parser.add_argument(
        "--ml_algo",
        default="classfication",
        type=str,
        help="classification or regression ? "
    )
    parser.add_argument(
        "--top",
        default="e",
        type=str,
        help="Extract top N electrodes(e) wise ranking, feature(f) wise ranking or both combined(ef)",
    )

    parser.add_argument(
        "--fs_method",
        default="SelectKBest",
        type=str,
        help="SKLearn feature selection method to be performed",
    )
    
    my_args = parser.parse_args()

    return my_args