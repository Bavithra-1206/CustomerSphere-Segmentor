import os
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np

from src.data_loader import CustomerDataLoader
from src.kmeans_engine import KMeansSegmentationEngine
from src.persona_generator import CustomerPersonaGenerator

# Global in-memory session cache for current active dataset and trained model
SESSION = {
    "loader": None,
    "engine": None,
    "current_dataset": "data/mall_customers.csv",
    "selected_features": [],
    "scaler_type": "standard",
    "model_fit_res": None,
    "personas": None
}

class CustomerSegmentationHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Concise logging
        print(f"[{self.command}] {self.path}")

    def send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            self.serve_file("templates/index.html", "text/html")
        elif path.startswith("/static/"):
            file_path = path.lstrip("/")
            ext = os.path.splitext(file_path)[1]
            mime_types = {
                ".css": "text/css",
                ".js": "application/javascript",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".svg": "image/svg+xml"
            }
            self.serve_file(file_path, mime_types.get(ext, "application/octet-stream"))
        elif path == "/api/datasets":
            datasets = [
                {
                    "id": "mall_customers",
                    "name": "Mall Customers Dataset",
                    "filename": "data/mall_customers.csv",
                    "description": "Standard benchmark with Age, Annual Income, and Spending Score."
                },
                {
                    "id": "ecommerce_rfm",
                    "name": "E-Commerce RFM + Demographics",
                    "filename": "data/ecommerce_rfm_demographics.csv",
                    "description": "Rich dataset with Recency, Frequency, Monetary, Income, Age, and Region."
                }
            ]
            self.send_json({"datasets": datasets})
        else:
            self.send_json({"error": "Not Found"}, status=404)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            body = {}

        if path == "/api/load_data":
            filename = body.get("filename", "data/mall_customers.csv")
            if not os.path.exists(filename):
                self.send_json({"error": f"File {filename} not found."}, status=404)
                return
            
            try:
                SESSION["loader"] = CustomerDataLoader(filepath=filename)
                SESSION["current_dataset"] = filename
                col_info = SESSION["loader"].get_column_info()
                sample_rows = SESSION["loader"].df.head(10).to_dict(orient="records")
                summary_stats = SESSION["loader"].get_summary_stats()

                # Clean float NaNs for JSON serialization
                cleaned_stats = {}
                for k, v in summary_stats.items():
                    cleaned_stats[k] = {col: round(float(val), 2) for col, val in v.items()}

                self.send_json({
                    "status": "success",
                    "filename": filename,
                    "info": col_info,
                    "sample_rows": sample_rows,
                    "summary_stats": cleaned_stats
                })
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)

        elif path == "/api/elbow_silhouette":
            selected_features = body.get("features", [])
            scaling_method = body.get("scaling", "standard")

            if not SESSION["loader"]:
                SESSION["loader"] = CustomerDataLoader(filepath="data/mall_customers.csv")

            scaled_df, feat_cols = SESSION["loader"].preprocess(
                selected_features=selected_features, 
                scaling_method=scaling_method
            )
            
            SESSION["selected_features"] = feat_cols
            SESSION["scaler_type"] = scaling_method
            
            engine = KMeansSegmentationEngine(scaled_df, SESSION["loader"].raw_df, feat_cols, scaler=SESSION["loader"].scaler)
            eval_results = engine.compute_elbow_and_silhouette(k_min=2, k_max=10)
            
            self.send_json(eval_results)

        elif path == "/api/cluster":
            n_clusters = int(body.get("n_clusters", 5))
            selected_features = body.get("features", [])
            scaling_method = body.get("scaling", "standard")

            if not SESSION["loader"]:
                SESSION["loader"] = CustomerDataLoader(filepath="data/mall_customers.csv")

            scaled_df, feat_cols = SESSION["loader"].preprocess(
                selected_features=selected_features, 
                scaling_method=scaling_method
            )

            SESSION["selected_features"] = feat_cols
            SESSION["scaler_type"] = scaling_method

            engine = KMeansSegmentationEngine(scaled_df, SESSION["loader"].raw_df, feat_cols, scaler=SESSION["loader"].scaler)
            fit_res = engine.fit(n_clusters=n_clusters)
            pca_data = engine.get_pca_projection(n_components=3)
            profiles = engine.get_cluster_profiles()
            personas = CustomerPersonaGenerator.generate_personas(profiles["profiles"], profiles["overall_means"])

            SESSION["engine"] = engine
            SESSION["model_fit_res"] = fit_res
            SESSION["personas"] = personas

            # Format clustered customer sample records for UI
            clustered_df = fit_res["clustered_df"]
            table_sample = clustered_df.head(50).to_dict(orient="records")

            self.send_json({
                "status": "success",
                "n_clusters": fit_res["n_clusters"],
                "inertia": fit_res["inertia"],
                "silhouette_score": fit_res["silhouette_score"],
                "calinski_harabasz_score": fit_res["calinski_harabasz_score"],
                "davies_bouldin_score": fit_res["davies_bouldin_score"],
                "pca": pca_data,
                "profiles": profiles,
                "personas": personas,
                "sample_table": table_sample,
                "feature_cols": feat_cols
            })

        elif path == "/api/predict":
            input_features = body.get("features", {})
            if not SESSION["engine"] or not SESSION["personas"]:
                self.send_json({"error": "Model must be trained first. Run clustering prior to prediction."}, status=400)
                return

            try:
                pred = SESSION["engine"].predict_sample(input_features)
                cluster_id = pred["predicted_cluster"]
                persona = SESSION["personas"].get(cluster_id, {})
                
                self.send_json({
                    "predicted_cluster": cluster_id,
                    "centroid_distance": pred["centroid_distance"],
                    "persona": persona
                })
            except Exception as e:
                self.send_json({"error": str(e)}, status=500)
        else:
            self.send_json({"error": "API route not found"}, status=404)

    def serve_file(self, rel_path, content_type):
        if not os.path.exists(rel_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        with open(rel_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

def run_server(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CustomerSegmentationHandler)
    print(f"[SERVER] Customer Segmentation Server running on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
