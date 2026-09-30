import pandas as pd
from src.data_loader import CustomerDataLoader
from src.kmeans_engine import KMeansSegmentationEngine
from src.persona_generator import CustomerPersonaGenerator

def run_test():
    print("--- Testing Customer Segmentation Pipeline ---")
    
    # 1. Load Data
    loader = CustomerDataLoader(filepath="data/mall_customers.csv")
    info = loader.get_column_info()
    print("Column Info:", info)
    
    # 2. Preprocess Data
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    scaled_df, feat_cols = loader.preprocess(selected_features=features, scaling_method="standard")
    
    # 3. K-Means Engine
    engine = KMeansSegmentationEngine(scaled_df, loader.raw_df, feat_cols, scaler=loader.scaler)
    
    # 4. Compute Elbow & Silhouette
    metrics = engine.compute_elbow_and_silhouette(k_min=2, k_max=8)
    print("Optimal K recommended:", metrics["recommended_k"])
    print("Silhouette Scores:", metrics["silhouette_scores"])
    
    # 5. Fit Model for K = 5
    fit_res = engine.fit(n_clusters=5)
    print(f"Fit Complete! Inertia: {fit_res['inertia']:.2f}, Silhouette: {fit_res['silhouette_score']:.4f}")
    
    # 6. PCA Projection
    pca_data = engine.get_pca_projection()
    print("PCA Explained Variance:", pca_data["explained_variance"])
    
    # 7. Cluster Profiles & Personas
    profiles = engine.get_cluster_profiles()
    personas = CustomerPersonaGenerator.generate_personas(profiles["profiles"], profiles["overall_means"])
    
    for c_id, persona in personas.items():
        print(f"\n[Cluster {c_id}] {persona['persona_title']} ({persona['customer_percentage']}%)")
        print("  Summary:", persona['demographic_summary'])
        print("  Strategy:", persona['recommended_strategy'])

    print("\n--- Pipeline Test PASSED Successfully ---")

if __name__ == "__main__":
    run_test()
