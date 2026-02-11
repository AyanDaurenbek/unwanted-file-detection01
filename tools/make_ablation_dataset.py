import pandas as pd

train = pd.read_csv("data/synthetic_unwanted_files_v3_train.csv")
val = pd.read_csv("data/synthetic_unwanted_files_v3_val.csv")

ablation = pd.concat([train, val], ignore_index=True)

ablation.to_csv(
    "data/synthetic_unwanted_files_v3_ablation.csv",
    index=False
)

print("OK: ablation dataset created")
print(ablation.shape)
