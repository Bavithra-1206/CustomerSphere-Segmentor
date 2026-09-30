import os
import pandas as pd
import numpy as np

def generate_mall_customers(filename="data/mall_customers.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.random.seed(42)
    
    n_per_cluster = [40, 35, 40, 35, 50] # 200 total
    clusters_data = []
    
    # Cluster 1: High Income, High Spending (VIP / High Rollers)
    income1 = np.random.normal(88, 12, n_per_cluster[0]).clip(70, 140)
    spending1 = np.random.normal(82, 9, n_per_cluster[0]).clip(65, 99)
    age1 = np.random.normal(32, 6, n_per_cluster[0]).clip(18, 50)
    gender1 = np.random.choice(["Female", "Male"], n_per_cluster[0], p=[0.55, 0.45])
    
    # Cluster 2: High Income, Low Spending (Savers / Conservative)
    income2 = np.random.normal(88, 14, n_per_cluster[1]).clip(70, 137)
    spending2 = np.random.normal(18, 9, n_per_cluster[1]).clip(1, 40)
    age2 = np.random.normal(41, 10, n_per_cluster[1]).clip(20, 70)
    gender2 = np.random.choice(["Female", "Male"], n_per_cluster[1], p=[0.48, 0.52])

    # Cluster 3: Low Income, High Spending (Impulse / Trend Seekers)
    income3 = np.random.normal(26, 7, n_per_cluster[2]).clip(15, 39)
    spending3 = np.random.normal(78, 10, n_per_cluster[2]).clip(60, 99)
    age3 = np.random.normal(25, 5, n_per_cluster[2]).clip(18, 38)
    gender3 = np.random.choice(["Female", "Male"], n_per_cluster[2], p=[0.60, 0.40])

    # Cluster 4: Low Income, Low Spending (Budget Conscious)
    income4 = np.random.normal(25, 6, n_per_cluster[3]).clip(15, 39)
    spending4 = np.random.normal(20, 8, n_per_cluster[3]).clip(1, 40)
    age4 = np.random.normal(45, 12, n_per_cluster[3]).clip(19, 68)
    gender4 = np.random.choice(["Female", "Male"], n_per_cluster[3], p=[0.50, 0.50])

    # Cluster 5: Average Income, Average Spending (Mainstream)
    income5 = np.random.normal(55, 8, n_per_cluster[4]).clip(40, 70)
    spending5 = np.random.normal(50, 7, n_per_cluster[4]).clip(38, 65)
    age5 = np.random.normal(38, 12, n_per_cluster[4]).clip(18, 65)
    gender5 = np.random.choice(["Female", "Male"], n_per_cluster[4], p=[0.58, 0.42])

    incomes = np.concatenate([income1, income2, income3, income4, income5])
    spendings = np.concatenate([spending1, spending2, spending3, spending4, spending5])
    ages = np.concatenate([age1, age2, age3, age4, age5])
    genders = np.concatenate([gender1, gender2, gender3, gender4, gender5])
    
    df = pd.DataFrame({
        "CustomerID": [1001 + i for i in range(len(incomes))],
        "Gender": genders,
        "Age": np.round(ages).astype(int),
        "Annual Income (k$)": np.round(incomes).astype(int),
        "Spending Score (1-100)": np.round(spendings).astype(int)
    })
    
    # Shuffle slightly so clusters aren't strictly contiguous
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["CustomerID"] = [1001 + i for i in range(len(df))]
    
    df.to_csv(filename, index=False)
    print(f"Generated Mall Customers dataset: {filename} ({len(df)} rows)")
    return df

def generate_ecommerce_rfm(filename="data/ecommerce_rfm_demographics.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.random.seed(101)
    
    n_samples = 450
    
    # Archetype 1: VIP Loyalists (High Income, High Frequency, Low Recency, High Monetary)
    n1 = 90
    age1 = np.random.normal(38, 8, n1).clip(22, 60)
    gender1 = np.random.choice(["Female", "Male", "Other"], n1, p=[0.52, 0.45, 0.03])
    income1 = np.random.normal(110, 20, n1).clip(75, 180)
    score1 = np.random.normal(88, 7, n1).clip(70, 100)
    recency1 = np.random.normal(12, 8, n1).clip(1, 35)
    frequency1 = np.random.normal(28, 6, n1).clip(15, 50)
    monetary1 = np.random.normal(4500, 800, n1).clip(2500, 8000)
    region1 = np.random.choice(["North America", "Europe", "Asia Pacific"], n1, p=[0.5, 0.3, 0.2])
    cat1 = np.random.choice(["Electronics", "Fashion", "Home & Luxury"], n1, p=[0.4, 0.4, 0.2])

    # Archetype 2: Seasonal High-Spenders (Moderate Recency, High Monetary, Low Frequency)
    n2 = 80
    age2 = np.random.normal(45, 10, n2).clip(25, 68)
    gender2 = np.random.choice(["Female", "Male", "Other"], n2, p=[0.45, 0.50, 0.05])
    income2 = np.random.normal(92, 15, n2).clip(60, 140)
    score2 = np.random.normal(65, 10, n2).clip(45, 85)
    recency2 = np.random.normal(65, 20, n2).clip(30, 120)
    frequency2 = np.random.normal(6, 2, n2).clip(2, 12)
    monetary2 = np.random.normal(2800, 600, n2).clip(1500, 5000)
    region2 = np.random.choice(["North America", "Europe", "Latin America"], n2, p=[0.4, 0.4, 0.2])
    cat2 = np.random.choice(["Home & Luxury", "Electronics", "Beauty"], n2, p=[0.5, 0.3, 0.2])

    # Archetype 3: Young Trend Hunters (Low/Mod Income, High Score, High Frequency, Low Monetary per Tx)
    n3 = 100
    age3 = np.random.normal(24, 4, n3).clip(18, 33)
    gender3 = np.random.choice(["Female", "Male", "Other"], n3, p=[0.60, 0.35, 0.05])
    income3 = np.random.normal(35, 8, n3).clip(18, 55)
    score3 = np.random.normal(82, 8, n3).clip(65, 98)
    recency3 = np.random.normal(18, 10, n3).clip(2, 45)
    frequency3 = np.random.normal(19, 5, n3).clip(8, 35)
    monetary3 = np.random.normal(1200, 300, n3).clip(500, 2200)
    region3 = np.random.choice(["North America", "Asia Pacific", "Europe"], n3, p=[0.4, 0.4, 0.2])
    cat3 = np.random.choice(["Fashion", "Beauty", "Electronics"], n3, p=[0.5, 0.35, 0.15])

    # Archetype 4: Churned / At-Risk Bargain Buyers (High Recency, Low Score, Low Frequency, Low Monetary)
    n4 = 90
    age4 = np.random.normal(52, 11, n4).clip(25, 72)
    gender4 = np.random.choice(["Female", "Male", "Other"], n4, p=[0.48, 0.48, 0.04])
    income4 = np.random.normal(42, 12, n4).clip(20, 75)
    score4 = np.random.normal(22, 9, n4).clip(1, 42)
    recency4 = np.random.normal(140, 35, n4).clip(80, 250)
    frequency4 = np.random.normal(3, 1.5, n4).clip(1, 7)
    monetary4 = np.random.normal(400, 150, n4).clip(80, 900)
    region4 = np.random.choice(["North America", "Europe", "Latin America"], n4, p=[0.35, 0.35, 0.30])
    cat4 = np.random.choice(["Books & Media", "Fashion", "Home & Luxury"], n4, p=[0.4, 0.3, 0.3])

    # Archetype 5: Steady Moderate Shoppers (Mid Income, Mid Recency, Mid Frequency, Mid Spend)
    n5 = 90
    age5 = np.random.normal(36, 9, n5).clip(20, 60)
    gender5 = np.random.choice(["Female", "Male", "Other"], n5, p=[0.50, 0.47, 0.03])
    income5 = np.random.normal(62, 10, n5).clip(45, 80)
    score5 = np.random.normal(52, 8, n5).clip(38, 68)
    recency5 = np.random.normal(40, 15, n5).clip(10, 80)
    frequency5 = np.random.normal(11, 3, n5).clip(5, 20)
    monetary5 = np.random.normal(1800, 400, n5).clip(900, 2900)
    region5 = np.random.choice(["North America", "Europe", "Asia Pacific"], n5, p=[0.4, 0.3, 0.3])
    cat5 = np.random.choice(["Electronics", "Fashion", "Beauty", "Home & Luxury"], n5, p=[0.3, 0.3, 0.2, 0.2])

    df = pd.DataFrame({
        "CustomerID": [5000 + i for i in range(n_samples)],
        "Age": np.round(np.concatenate([age1, age2, age3, age4, age5])).astype(int),
        "Gender": np.concatenate([gender1, gender2, gender3, gender4, gender5]),
        "Annual_Income_k": np.round(np.concatenate([income1, income2, income3, income4, income5]), 1),
        "Spending_Score": np.round(np.concatenate([score1, score2, score3, score4, score5])).astype(int),
        "Recency_Days": np.round(np.concatenate([recency1, recency2, recency3, recency4, recency5])).astype(int),
        "Frequency_Purchases": np.round(np.concatenate([frequency1, frequency2, frequency3, frequency4, frequency5])).astype(int),
        "Monetary_Value_USD": np.round(np.concatenate([monetary1, monetary2, monetary3, monetary4, monetary5]), 2),
        "Region": np.concatenate([region1, region2, region3, region4, region5]),
        "Preferred_Category": np.concatenate([cat1, cat2, cat3, cat4, cat5])
    })
    
    df = df.sample(frac=1, random_state=101).reset_index(drop=True)
    df["CustomerID"] = [5000 + i for i in range(len(df))]
    
    df.to_csv(filename, index=False)
    print(f"Generated E-commerce RFM dataset: {filename} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    generate_mall_customers()
    generate_ecommerce_rfm()
