import argparse
from pykeen.hpo import hpo_pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="pykeen_hpo",  # プログラム名
        usage="python script_pykeen_hpo.py <model> <dataset> <n_trials> <output_dir>", # プログラムの利用方法
        description="script for hyper parameter optimization of knowledge graph embedding with pykeen", # ヘルプの前に表示
        epilog="end", # ヘルプの後に表示
        add_help=True, # -h/–-helpオプションの追加
    )

    parser.add_argument("-m", "--model", type=str, help="model")
    parser.add_argument("-d", "--dataset", type=str, help="dataset")
    parser.add_argument("-n", "--n_trials", type=int, help="number of trials for optimizaion")
    parser.add_argument("-o", "--output", type=str, help="output directory")

    args = parser.parse_args()

    hpo_pipeline_result = hpo_pipeline(
        n_trials=args.n_trials,
        dataset=args.dataset,
        model=args.model,
    )
    
    hpo_pipeline_result.save_to_directory(args.output)