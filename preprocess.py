import pandas as pd

def load_data():
    df = pd.read_csv(
        "household_power_consumption.txt",
        sep=";",
        na_values="?",
        low_memory=False
    )

    df.dropna(inplace=True)

    cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]

    # Convert safely to numeric
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    # Reduce dataset size
    df = df.head(10000)

    return df