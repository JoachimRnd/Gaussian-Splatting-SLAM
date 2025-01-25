import pandas as pd
import glob
import os
import shutil


def combine_results():
    eval_dir = "evaluation_metrics"
    raw_dir = os.path.join(eval_dir, "raw_results")
    summary_dir = os.path.join(eval_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(raw_dir, "*_metrics.csv"))

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    mean_values = combined_df[combined_df["optimization"] == "after"].mean()
    mean_row = pd.DataFrame(
        [
            {
                "test_name": "GLOBAL_AVERAGE",
                "optimization": "after",
                "psnr": mean_values["psnr"],
                "ssim": mean_values["ssim"],
                "lpips": mean_values["lpips"],
                "ate": mean_values["ate"],
                "fps": mean_values["fps"],
            }
        ]
    )
    final_df = pd.concat([combined_df, mean_row], ignore_index=True)
    final_df.to_csv(os.path.join(summary_dir, "combined_results.csv"), index=False)

    shutil.rmtree(raw_dir)


if __name__ == "__main__":
    combine_results()
