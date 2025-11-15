import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns


# Convert .data + .names into csv
def convert_to_csv(data_path="data/breast-cancer-wisconsin.data"):
    # Define the 11 column names (from the .names file)
    columns = [
        "Id",
        "Clump_Thickness",
        "Uniformity_Cell_Size",
        "Uniformity_Cell_Shape",
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size",
        "Bare_Nuclei",
        "Bland_Chromatin",
        "Normal_Nucleoli",
        "Mitoses",
        "Class"
    ]

    # Load the .data file (comma-separated)
    df = pd.read_csv(data_path, header=None, names=columns)

    # Save as CSV
    df.to_csv("data/data.csv", index=False)

def load_data(path="data/data.csv"):
    df = pd.read_csv(path)

    ## Data cleanup
    # Removing ’Id’ column
    df = df.drop(columns=['Id'])

    # Dropping rows with missing values
    df = df[df["Bare_Nuclei"] != "?"]

    # Updating data types
    df["Bare_Nuclei"] = df["Bare_Nuclei"].astype(int)
    df["Class"] = df["Class"].astype(int)

    # Converting ’Class’ values to binary
    df["Class"] = df["Class"].map({2: 0, 4: 1})

    # Balancing the dataset
    benign = df[df["Class"] == 0]
    malignant = df[df["Class"] == 1]

    if len(benign) > len(malignant):
        malignant_upsampled = resample(
            malignant,
            replace=True,
            n_samples=len(benign),
            random_state=42
        )
        df_balanced = pd.concat([benign, malignant_upsampled])
    else:
        benign_upsampled = resample(
            benign,
            replace=True,
            n_samples=len(malignant),
            random_state=42
        )
        df_balanced = pd.concat([benign_upsampled, malignant])

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


    ## Exploration and Visualization
    # Parallel Coordinate Plot
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df.drop('Class', axis=1))
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    df_scaled['Class'] = df['Class']

    # Convert class to label names for color legend
    df_scaled['Class'] = df_scaled['Class'].map({0: 'Benign', 1: 'Malignant'})

    # Create the parallel coordinates plot
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df_scaled, 'Class', color=['#1f77b4', '#d62728'], alpha=0.4)
    plt.title("Parallel Coordinate Plot - Breast Cancer Wisconsin (Original)")
    plt.xlabel("Features")
    plt.ylabel("Scaled Feature Values")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()

    # Distribution of Each Feature for Both Classes
    sns.set(style="whitegrid")

    # Create a grid of subplots (3x3 for 9 features)
    features = df.columns[:-1]  # exclude 'Class'
    plt.figure(figsize=(15, 12))

    for i, feature in enumerate(features):
        plt.subplot(3, 3, i + 1)
        sns.kdeplot(data=df, x=feature, hue="Class", fill=True, common_norm=False, palette={0: "#1f77b4", 1: "#d62728"})
        plt.title(feature)
        plt.xlabel('')
        plt.ylabel('Density')

    plt.suptitle("Distribution of Each Feature for Benign and Malignant Samples", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return df_balanced