import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PLOT_FOLDER = "plots"


# Function to generate checkpoints (linear or non-linear)
def generate_checkpoints(total_rows, growth_type="linear", growth_rate=1.1):
    checkpoints = []
    if growth_type == "linear":
        checkpoints = list(range(1, total_rows + 1))
    elif growth_type == "nonlinear":
        i = 1
        while True:
            checkpoint = min(int(i), total_rows)
            checkpoints.append(checkpoint)
            if checkpoint >= total_rows:
                break
            i *= growth_rate
    return checkpoints


# Function to produce and save charts
def analyze_and_plot(dataset_path, title):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Ensure relevant columns are numeric
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce")

    # Define parameters for plotting
    parameters = ["age", "income"]  # Columns to analyze

    # Generate checkpoints
    growth_type = "nonlinear"  # Options: 'linear' or 'nonlinear'
    checkpoints = generate_checkpoints(
        len(df), growth_type=growth_type, growth_rate=1.02
    )

    # Create plots for each parameter
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    for idx, parameter in enumerate(parameters):
        if parameter not in df.columns or not np.issubdtype(
            df[parameter].dtype, np.number
        ):
            continue

        medians = []
        for checkpoint in checkpoints:
            median_value = df[parameter].iloc[:checkpoint].mean()
            medians.append(median_value)

        # Calculate the dataset-wide mean and 1% deviation borders
        dataset_mean = df[parameter].mean()
        upper_border = dataset_mean * 1.01
        lower_border = dataset_mean * 0.99

        # Find the last checkpoint where the difference is more than 1%
        vertical_line_position = None
        for i in range(len(medians) - 1, -1, -1):
            if not (lower_border <= medians[i] <= upper_border):
                vertical_line_position = checkpoints[i]
                break

        axes[idx].plot(checkpoints, medians, label="Median Convergence")
        axes[idx].axhline(
            dataset_mean, color="red", linestyle="-", label="Dataset Mean"
        )
        axes[idx].axhline(
            upper_border, color="orange", linestyle="--", label="1% Upper Border"
        )
        axes[idx].axhline(
            lower_border, color="orange", linestyle="--", label="1% Lower Border"
        )

        # Add vertical dotted line if a position is found
        if vertical_line_position is not None:
            axes[idx].axvline(
                vertical_line_position,
                color="green",
                linestyle="dotted",
                label=f"{vertical_line_position} samples",
            )

        axes[idx].set_title(f"{parameter.capitalize()} Convergence")
        axes[idx].set_xlabel("Number of Samples")
        axes[idx].set_ylabel(f"Mean {parameter.capitalize()}")
        axes[idx].legend()
        axes[idx].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{PLOT_FOLDER}/{title.replace(' ', '_').lower()}_convergence.png")
    plt.show()


# Main execution
if __name__ == "__main__":
    analyze_and_plot("gpt_with_data_personas.csv", "GPT enhanced with statistical data")
    analyze_and_plot("gpt_plain.csv", "GPT without additional data")
    # Uncomment if needed: analyze_and_plot("clones_24.csv", "Clones 24 Dataset")
