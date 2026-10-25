import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
from pathlib import Path

df = pd.read_csv("data/academic_survival_longitudinal.csv")

display(df.head())
print("Dimensions:", df.shape)
print("Unique students:", df["Student_ID"].nunique())


df.info()
print()
display(df.isna().sum())

print("Duplicate rows:", df.duplicated().sum())
print(
    "Duplicate student-semester pairs:",
    df.duplicated(["Student_ID", "Semester"]).sum()
)
print("Censored records:", df["Censored"].eq(1).sum())

display(df["Target_Dropout_Next_Sem"].value_counts())


df.columns = df.columns.str.strip()

text_columns = [
    "Student_ID",
    "Gender",
    "Housing_Status",
    "End_of_Semester_Status"
]

for column in text_columns:
    df[column] = df[column].str.strip()

numeric_columns = [
    column for column in df.columns
    if column not in text_columns
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="raise")

exact_duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

duplicate_keys = int(
    df.duplicated(["Student_ID", "Semester"]).sum()
)

if duplicate_keys:
    raise ValueError(
        "Repeated student-semester pairs found. "
        "Check these records before continuing."
    )

df["Gender"] = df["Gender"].replace({
    "F": "Female",
    "female": "Female",
    "M": "Male",
    "male": "Male"
})

df["Dropout"] = df["Target_Dropout_Next_Sem"]


# Check attendance before adjusting values outside 0–100
attendance_below_zero = int((df["Attendance"] < 0).sum())
attendance_above_100 = int((df["Attendance"] > 100).sum())

print("Attendance below 0:", attendance_below_zero)
print("Attendance above 100:", attendance_above_100)

df["Attendance"] = df["Attendance"].clip(0, 100)

overall_dropout_rate = df["Dropout"].mean()

print("Dropout-labeled records:", f"{overall_dropout_rate:.2%}")


output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
RED = "#e34948"
GRID = "#e3e2de"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

bootstrap_samples = 400
random_seed = 0

stats = {
    "n_rows": int(len(df)),
    "n_students": int(df["Student_ID"].nunique()),
    "row_dropout_rate": float(overall_dropout_rate),
    "censored_records": int(df["Censored"].eq(1).sum()),
    "exact_duplicates_removed": exact_duplicates,
    "duplicate_student_semester_keys": duplicate_keys,
    "attendance_below_zero": attendance_below_zero,
    "attendance_above_100": attendance_above_100
}


def save_and_show(figure, filename):
    figure.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(figure)


def bootstrap_dropout_rates(data, group_column):
    student_ids = data["Student_ID"].unique()

    totals = (
        data.groupby(
            ["Student_ID", group_column],
            observed=True
        )["Dropout"]
        .agg(dropouts="sum", records="count")
        .reset_index()
    )

    dropout_counts = totals.pivot(
        index="Student_ID",
        columns=group_column,
        values="dropouts"
    ).reindex(student_ids).fillna(0)

    record_counts = totals.pivot(
        index="Student_ID",
        columns=group_column,
        values="records"
    ).reindex(student_ids).fillna(0)

    positives = dropout_counts.to_numpy()
    records = record_counts.to_numpy()

    group_rates = positives.sum(axis=0) / records.sum(axis=0)

    rng = np.random.default_rng(random_seed)
    sampled_rates = []

    for _ in range(bootstrap_samples):
        sampled_students = rng.integers(
            0,
            len(student_ids),
            size=len(student_ids)
        )

        sampled_positives = positives[sampled_students].sum(axis=0)
        sampled_records = records[sampled_students].sum(axis=0)

        rates = np.divide(
            sampled_positives,
            sampled_records,
            out=np.full(sampled_positives.shape, np.nan),
            where=sampled_records > 0
        )

        sampled_rates.append(rates)

    lower, upper = np.nanpercentile(
        sampled_rates,
        [2.5, 97.5],
        axis=0
    )

    return pd.DataFrame({
        "group": dropout_counts.columns,
        "rate": group_rates,
        "lo": lower,
        "hi": upper,
        "n_rows": records.sum(axis=0).astype(int)
    })


def plot_dropout_rates(
    axis,
    data,
    group_column,
    title,
    y_limit,
    show_y_labels=False
):
    rates = bootstrap_dropout_rates(data, group_column)

    positions = np.arange(len(rates))
    percentages = rates["rate"] * 100

    error_bars = [
        (rates["rate"] - rates["lo"]) * 100,
        (rates["hi"] - rates["rate"]) * 100
    ]

    axis.bar(
        positions,
        percentages,
        width=0.62,
        color=BLUE,
        zorder=2
    )

    axis.errorbar(
        positions,
        percentages,
        yerr=error_bars,
        fmt="none",
        ecolor="dimgray",
        capsize=2.2,
        zorder=3
    )

    for position, percentage, upper in zip(
        positions,
        percentages,
        rates["hi"] * 100
    ):
        axis.text(
            position,
            upper + y_limit * 0.02,
            f"{percentage:.1f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    axis.axhline(
        overall_dropout_rate * 100,
        color="dimgray",
        linestyle="--",
        linewidth=0.8,
        label=f"Overall: {overall_dropout_rate:.1%}"
    )

    axis.set_xticks(
        positions,
        labels=rates["group"].astype(str)
    )

    axis.set_ylim(0, y_limit)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.yaxis.grid(True, color=GRID)
    axis.set_axisbelow(True)
    axis.tick_params(length=0, labelleft=show_y_labels)
    axis.spines["left"].set_visible(False)

    return rates


last_records = (
    df.sort_values("Semester")
    .groupby("Student_ID")
    .tail(1)
)

status_names = {
    "Enrolled": "Still enrolled\nat last observation",
    "Dropped_Out": "Dropped out",
    "Graduated": "Graduated"
}

outcome_counts = (
    last_records["End_of_Semester_Status"]
    .map(status_names)
    .value_counts()
    .reindex(list(status_names.values()), fill_value=0)
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 3.8),
    layout="constrained"
)

positions = np.arange(len(outcome_counts))

axes[0].barh(
    positions,
    outcome_counts,
    color=[BLUE, ORANGE, GREEN]
)

for position, count in zip(positions, outcome_counts):
    percentage = count / len(last_records)

    axes[0].text(
        count + 180,
        position,
        f"{count:,} ({percentage:.1%})",
        va="center"
    )

axes[0].set_yticks(positions, labels=outcome_counts.index)
axes[0].invert_yaxis()
axes[0].set_xlim(0, outcome_counts.max() * 1.55)
axes[0].set_xlabel("Number of students")
axes[0].set_title(
    "(a) Last observed student status",
    loc="left",
    fontweight="bold"
)

semester_rates = plot_dropout_rates(
    axes[1],
    df,
    "Semester",
    "(b) Dropout labels by semester",
    14,
    show_y_labels=True
)

axes[1].set_xlabel("Semester")
axes[1].set_ylabel("Dropout-labeled records (%)")
axes[1].legend(loc="upper left", frameon=False)

save_and_show(fig, "fig1_outcomes.png")

display(semester_rates)

stats["student_outcomes"] = {
    label.replace("\n", " "): int(count)
    for label, count in outcome_counts.items()
}

stats["student_outcomes_pct"] = {
    label.replace("\n", " "): float(count / len(last_records) * 100)
    for label, count in outcome_counts.items()
}

stats["dropout_by_semester"] = dict(zip(
    semester_rates["group"].astype(int),
    semester_rates["rate"]
))

stats["rows_by_semester"] = dict(zip(
    semester_rates["group"].astype(int),
    semester_rates["n_rows"].astype(int)
))

demographic_data = df.copy()

demographic_data["Income group"] = pd.qcut(
    demographic_data["Family_Income"],
    4,
    labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
)

demographic_data["First generation"] = (
    demographic_data["First_Generation"].map({0: "No", 1: "Yes"})
)

demographic_data["Scholarship group"] = (
    demographic_data["Scholarship"].map({0: "No", 1: "Yes"})
)

demographic_data["Financial stress group"] = pd.cut(
    demographic_data["Financial_Stress"],
    bins=[0, 2.1, 3.4, np.inf],
    labels=["Low\n(≤2.1)", "Mid\n(2.1–3.4)", "High\n(>3.4)"]
)

fig, axes = plt.subplots(
    1, 4,
    figsize=(12, 3.6),
    sharey=True,
    layout="constrained"
)

comparisons = [
    ("income", "Income group", "(a) Family income"),
    ("firstgen", "First generation", "(b) First generation"),
    ("scholarship", "Scholarship group", "(c) Scholarship"),
    ("stress", "Financial stress group", "(d) Financial stress")
]

socioeconomic_results = {}

for index, (name, column, title) in enumerate(comparisons):
    socioeconomic_results[name] = plot_dropout_rates(
        axes[index],
        demographic_data,
        column,
        title,
        21,
        show_y_labels=(index == 0)
    )

axes[0].set_ylabel("Dropout-labeled records (%)")
axes[0].legend(loc="upper left", frameon=False, fontsize=8)

save_and_show(fig, "fig2_socioeconomic.png")

stats["fig2"] = {}

for name, rates in socioeconomic_results.items():
    print(name)
    display(rates)
    stats["fig2"][name] = rates.round(4).to_dict("records")


demographic_data["Work hours group"] = pd.cut(
    demographic_data["Work_Hours"],
    bins=[-1, 10, 20, 30, 40],
    labels=["0–10", ">10–20", ">20–30", ">30–40"]
)

demographic_data["Age group"] = pd.cut(
    demographic_data["Age"],
    bins=[16, 18, 20, 23, np.inf],
    labels=["17–18", "19–20", "21–23", "24+"]
)

demographic_data["Household group"] = pd.cut(
    demographic_data["Household_Size"],
    bins=[0, 2, 4, 6],
    labels=["1–2", "3–4", "5–6"]
)

extra_columns = [
    "Gender",
    "Housing_Status",
    "Age group",
    "Work hours group",
    "Household group",
    "Tuition_Base"
]

stats["extra_groups"] = {}

for column in extra_columns:
    rates = bootstrap_dropout_rates(demographic_data, column)

    print(column)
    display(rates)

    stats["extra_groups"][column] = rates.round(4).to_dict("records")


academic_data = df.copy()

academic_data["GPA group"] = pd.cut(
    academic_data["Sem_GPA"],
    bins=[0, 2, 2.5, 3, 3.5, np.inf],
    right=False,
    labels=["<2.0", "2.0–<2.5", "2.5–<3.0", "3.0–<3.5", "≥3.5"]
)

academic_data["Failed courses group"] = (
    academic_data["Failed_Courses"]
    .clip(upper=4)
    .map({0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"})
)

academic_data["Attendance group"] = pd.cut(
    academic_data["Attendance"],
    bins=[0, 75, 80, 85, 90, np.inf],
    right=False,
    labels=["<75", "75–<80", "80–<85", "85–<90", "≥90"]
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 3.8),
    sharey=True,
    layout="constrained"
)

comparisons = [
    ("gpa", "GPA group", "(a) Semester GPA"),
    ("failed", "Failed courses group", "(b) Failed courses"),
    ("att", "Attendance group", "(c) Attendance (%)")
]

academic_results = {}

for index, (name, column, title) in enumerate(comparisons):
    academic_results[name] = plot_dropout_rates(
        axes[index],
        academic_data,
        column,
        title,
        62,
        show_y_labels=(index == 0)
    )

axes[0].set_ylabel("Dropout-labeled records (%)")
axes[2].legend(loc="upper right", frameon=False, fontsize=8)

save_and_show(fig, "fig3_academic.png")

stats["fig3"] = {}

for name, rates in academic_results.items():
    print(name)
    display(rates)
    stats["fig3"][name] = rates.round(4).to_dict("records")


academic_data["LMS quartile"] = pd.qcut(
    academic_data["LMS_Logins"],
    4
)

lms_rates = bootstrap_dropout_rates(
    academic_data,
    "LMS quartile"
)

lms_rates["group"] = lms_rates["group"].astype(str)

display(lms_rates)

stats["lms_quartiles"] = lms_rates.round(4).to_dict("records")


correlation_columns = [
    "Sem_GPA",
    "Attendance",
    "LMS_Logins",
    "Failed_Courses",
    "Advising_Visits",
    "Course_Load",
    "Work_Hours",
    "Financial_Stress",
    "Family_Income",
    "Dropout"
]

plot_labels = [
    "Semester GPA",
    "Attendance",
    "LMS logins",
    "Failed courses",
    "Advising visits",
    "Course load",
    "Work hours",
    "Financial stress",
    "Family income",
    "Dropout label"
]

correlations = df[correlation_columns].corr(method="spearman")

upper_triangle = np.triu(
    np.ones(correlations.shape, dtype=bool)
)

visible_correlations = np.ma.array(
    correlations.to_numpy(),
    mask=upper_triangle
)

color_map = LinearSegmentedColormap.from_list(
    "correlations",
    [BLUE, "#f0efec", RED]
)

fig, axis = plt.subplots(
    figsize=(8, 6),
    layout="constrained"
)

image = axis.imshow(
    visible_correlations,
    cmap=color_map,
    vmin=-0.6,
    vmax=0.6
)

for row in range(len(correlation_columns)):
    for column in range(row):
        value = correlations.iloc[row, column]

        axis.text(
            column,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > 0.4 else "black"
        )

axis.set_xticks(
    range(len(plot_labels)),
    labels=plot_labels,
    rotation=45,
    ha="right"
)

axis.set_yticks(
    range(len(plot_labels)),
    labels=plot_labels
)

axis.tick_params(length=0)

for spine in axis.spines.values():
    spine.set_visible(False)

color_bar = fig.colorbar(
    image,
    ax=axis,
    fraction=0.04,
    pad=0.02,
    extend="both"
)

color_bar.set_label("Spearman correlation")
color_bar.outline.set_visible(False)

save_and_show(fig, "fig4_correlations.png")

dropout_correlations = (
    correlations["Dropout"]
    .drop("Dropout")
    .sort_values()
)

display(dropout_correlations)

stats["spearman_with_dropout"] = (
    correlations["Dropout"]
    .drop("Dropout")
    .round(3)
    .to_dict()
)

stats["spearman_matrix"] = correlations.round(3).to_dict()


# Save the numerical results
with (output_dir / "eda_stats.json").open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        stats,
        file,
        indent=2,
        default=str,
        ensure_ascii=False
    )

print("Saved four figures and the summary statistics to:")
print(output_dir.resolve())