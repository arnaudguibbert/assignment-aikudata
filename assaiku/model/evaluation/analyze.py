import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_data(data_dict: dict, idx_class: int):
    data = pd.DataFrame.from_records(data_dict)
    data = pd.melt(
        data,
        id_vars=["class", "model", "set", "repetition"],
        value_name="score",
        var_name="metric",
    )
    data = data[data["class"] == idx_class]

    train_perf, test_perf = data[data.set == "train"], data[data.set == "test"]

    fig, (ax_train, ax_test) = plt.subplots(
        1, 2, figsize=(20, 10), sharey=False
    )

    min_value = min(train_perf.score.min(), test_perf.score.min())
    low_lim = max(0, min_value - 0.1)

    sns.barplot(
        data=train_perf, x="model", y="score", hue="metric", ax=ax_train
    )
    sns.barplot(data=test_perf, x="model", y="score", hue="metric", ax=ax_test)
    ax_train.tick_params("x", rotation=60)
    ax_test.tick_params("x", rotation=60)
    ax_train.set_ylim([low_lim, 1])
    ax_test.set_ylim([low_lim, 1])

    fig.tight_layout()
