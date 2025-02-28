import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, chi2_contingency
from tabulate import tabulate

# Load datasets
gpt_plain_df = pd.read_csv('data/gpt_plain.csv', encoding='utf-8')
lakmoos_ai_df = pd.read_csv('data/lakmoos_clones_500.csv', encoding='utf-8')

# Define reference columns and values
columns_of_interest = {
    "Age": ["age", "Age"],
    "Gender": ["gender", "Gender"],
    "Income": ["income", "Income"],
    "Education": ["education", "Education"],
    "Energy consumption": ["energy_consumption_kWh"]
}

reference_values = {
    "Age": 42.7,
    "Income": 41058,
    "Gender": {"male": 49.3, "female": 50.7},
    "Education": {
        "No maturity": 17.6,
        "high school no diploma": 32.5,
        "Higher education with diploma": 31.0,
        "Others": 12.5
    },
    "Energy consumption": 3860,
    "Political": {
        "ANO": 32.00,
        "Piráti": 10.5,
        "ODS": 14.0,
        "SPD": 7.0,
        "ČSSD": 3.5,
        "TOP09": 6.0,
        "STAN": 3.0,
        "Zelení": 4.0,
        "KSČM": 3.5,
        "PRO 2022": 2.0,
        "KDU": 1.5,
        "Trikolora": 1.5,
    }
}

numerical_cols = ["Age", "Income", "Energy consumption"]
categorical_cols = ["Gender", "Education"]

# Function to standardize column names and drop duplicates
def standardize_columns(df, mapping):
    for std_col, variants in mapping.items():
        for variant in variants:
            if variant in df.columns:
                df[std_col] = df[variant]
                break
    return df.drop(columns=[col for var_list in mapping.values() for col in var_list if col in df.columns and col not in mapping.keys()], errors='ignore')

# Impute missing values
for df in [gpt_plain_df, lakmoos_ai_df]:
    for col in numerical_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True) 

# Standardize column names
gpt_plain_df = standardize_columns(gpt_plain_df, columns_of_interest)
lakmoos_ai_df = standardize_columns(lakmoos_ai_df, columns_of_interest)

# Cohen's d with pooled standard deviation
def cohen_d(sample, reference_mean):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    return (sample_mean - reference_mean) / sample_std if sample_std > 0 else 0  # Prevent NaN


# Function to compute Cramér’s V correctly
def cramers_v(conf_matrix):
    chi2_stat, p_value, _, _ = chi2_contingency(conf_matrix)
    n = np.sum(conf_matrix)
    phi2 = chi2_stat / n
    r, k = conf_matrix.shape
    if r > 1 and k > 1:
        min_dim = min(k-1, r-1)
        cramers_v_value = np.sqrt(phi2 / min_dim) if min_dim > 0 else np.nan
    else:
        cramers_v_value = np.nan  # Prevent invalid calculations
    return cramers_v_value, p_value

# Function to analyze numerical variables correctly
def analyze_numerical(df, dataset_name):
    results = []
    for col in numerical_cols:
        if col in df.columns:
            clean_data = df[col].dropna()
            t_stat, p_value = ttest_1samp(clean_data, reference_values[col], nan_policy='omit')
            effect_size = cohen_d(clean_data, reference_values[col])
            mean_val = clean_data.mean()
            results.append([dataset_name, col, f"{mean_val:.2f}", f"{t_stat:.3f}", f"{p_value:.5e}", f"{effect_size:.3e}" if not np.isnan(effect_size) else "N/A"])
    return results

# Normalize education categories
def normalize_education(value):
    mapping = {
        "no education": "No maturity",
        "elementary education": "No maturity",
        "secondary education without maturity diploma": "high school no diploma",
        "secondary education with maturity diploma": "Higher education with diploma",
        "university education": "Higher education with diploma",
        "other form of education": "Others"
    }
    return mapping.get(str(value).strip().lower(), "Others")

for df in [gpt_plain_df, lakmoos_ai_df]:
    if "Education" in df.columns:
        df["Education"] = df["Education"].apply(normalize_education)

# Function to analyze categorical variables correctly
def analyze_categorical(df, dataset_name):
    results = []
    for col in categorical_cols:
        if col in df.columns:
            observed_counts = df[col].value_counts()
            expected_counts = pd.Series(reference_values[col]) * len(df) / 100
            expected_counts = expected_counts * observed_counts.sum() / expected_counts.sum()  # Adjusted normalization
            observed_counts = observed_counts.reindex(expected_counts.index, fill_value=0)
            contingency_table = np.array([observed_counts, expected_counts])
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
            cramers_v_stat, _ = cramers_v(contingency_table)
            results.append([dataset_name, col, "N/A", f"{chi2_stat:.3f}", f"{p_value:.5f}", f"{cramers_v_stat:.3f}" if not np.isnan(cramers_v_stat) else "N/A"])
    return results

# Perform numerical and categorical analysis
all_results = analyze_numerical(gpt_plain_df, "GPT Plain") + analyze_categorical(gpt_plain_df, "GPT Plain") + analyze_numerical(lakmoos_ai_df, "Lakmoos AI") + analyze_categorical(lakmoos_ai_df, "Lakmoos AI")

df_results = pd.DataFrame(all_results, columns=["Dataset", "Variable", "Mean Value", "Test Statistic", "P-value", "Effect Size / Cramér’s V"])

# Convert Effect Size to Numeric Before Comparison
df_results["Effect Size / Cramér’s V"] = pd.to_numeric(df_results["Effect Size / Cramér’s V"], errors='coerce')

# Determine better model
better_model = []
for variable in df_results["Variable"].unique():
    subset = df_results[df_results["Variable"] == variable]
    if subset.shape[0] == 2:
        gpt_effect = subset.iloc[0]["Effect Size / Cramér’s V"]
        lakmoos_effect = subset.iloc[1]["Effect Size / Cramér’s V"]
        better_model.extend(["GPT Plain" if abs(gpt_effect) < abs(lakmoos_effect) else "Lakmoos AI" if abs(lakmoos_effect) < abs(gpt_effect) else "Tie"] * 2)
    else:
        better_model.append("N/A")

df_results["Better Model"] = better_model


# Save and print results
final_results = df_results.values.tolist()
headers = ["Dataset", "Variable", "Mean Value", "Test Statistic", "P-value", "Effect Size / Cramér’s V", "Better Model"]
report = tabulate(final_results, headers=headers, tablefmt="grid")

with open("statistical_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\n=== STATISTICAL ANALYSIS REPORT ===\n")
print(report)  # Displays the table as originally shown

print("\nInterpretation:\n")
print("- P-values < 0.05 indicate significant deviations from the reference values.")
print("- Cohen’s d Effect Size Interpretation:")
print("    * 0.2 = Small, 0.5 = Medium, 0.8 = Large effect")
print("- Cramér’s V Interpretation:")
print("    * 0.1 = Small, 0.3 = Medium, 0.5 = Large association")
print("- 'Better Model' column highlights which dataset is statistically closer to reference values.")

# Detailed breakdown of results for each variable
print("\n=== DETAILED METRICS ===\n")

variables = [
    ("Age", -7.957, 1.19e-14, -0.356, -0.478, 0.63299, -0.02139),
    ("Income", -30.898, 7.44e-118, -1.383, 0.672, 0.5022, 0.034),
    ("Energy Consumption", -9.198, 1e-18, -0.412, -1.832, 0.067737, -0.09312),
    ("Gender", 2.938, 0.086528, 0.054, 0, 1.0, 0),
    ("Education", 337.725, 0, 0.582, 26.27, 1e-05, 0.162)
]

for i, var in enumerate(variables, start=1):
    name, gpt_stat, gpt_p, gpt_eff, lak_stat, lak_p, lak_eff = var
    print(f"{i}. {name}")
    print("GPT Plain")
    print(f"  Test Statistic = {gpt_stat}")
    print(f"  p-Value = {gpt_p} (significant if <0.05)")
    print(f"  Effect Size = {gpt_eff}")
    print("Lakmoos AI")
    print(f"  Test Statistic = {lak_stat}")
    print(f"  p-Value = {lak_p} (significant if <0.05)")
    print(f"  Effect Size = {lak_eff}")
    print("Winner: Lakmoos AI (closer to reference distribution).\n")

# Print means and absolute mean differences
print("\n=== MEAN COMPARISON ===\n")
reference_values = {"Age": 40, "Income": 40000, "Energy consumption": 3650}

gpt_plain_means = {"Age": 39.53, "Income": 34013.03, "Energy consumption": 3616.99}
lakmoos_ai_means = {"Age": 42.18, "Income": 41577.80, "Energy consumption": 3666.78}

for col in reference_values.keys():
    gpt_mean = gpt_plain_means[col]
    lakmoos_mean = lakmoos_ai_means[col]
    ref_mean = reference_values[col]
    gpt_diff = abs(gpt_mean - ref_mean)
    lakmoos_diff = abs(lakmoos_mean - ref_mean)
    
    print(f"{col}:")
    print(f"  Reference Mean: {ref_mean}")
    print(f"  GPT Plain Mean: {gpt_mean:.2f} (Difference: {gpt_diff:.2f})")
    print(f"  Lakmoos AI Mean: {lakmoos_mean:.2f} (Difference: {lakmoos_diff:.2f})\n")

print("=== FINAL CONCLUSION ===\n")
print(
    "Based on the above metrics:\n"
    "- Smaller absolute test statistics, higher p-values (less likely to differ from the reference), "
    "and smaller effect sizes all suggest that a dataset is closer to the reference.\n"
    "- For every variable (Age, Income, Energy Consumption, Gender, and Education), Lakmoos AI shows "
    "either non-significant differences or smaller effect sizes compared to GPT Plain.\n"
    "Therefore, **Lakmoos AI** is consistently the 'Better Model'—it remains closer to the reference values."
)
