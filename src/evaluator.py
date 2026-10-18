import pandas as pd
import numpy as np
import math
from collections import defaultdict

class ModelEvaluator:
    def __init__(self, k=10):
        """
        Initializes the evaluator with the default K for Top-K metrics.
        """
        self.k = k

    def train_test_split(self, df, test_ratio=0.2, random_state=42):
        """
        Performs a user-centric train/test split. 
        Only positive interactions (attracted=1) are placed into the test set.
        Negative samples remain in the training set to prevent data leakage 
        and provide negative signals during training.
        """
        print("Starting User-centric Train/Test Split...")
        
        # Separate positives and negatives
        positives = df[df['attracted'] == 1].copy()
        negatives = df[df['attracted'] == 0].copy()
        
        # Filter users who have more than 1 positive interaction
        user_counts = positives['user_id'].value_counts()
        valid_users = user_counts[user_counts > 1].index
        valid_positives = positives[positives['user_id'].isin(valid_users)]
        
        # Sample 'test_ratio' of positive items per valid user for the test set
        test_positives = valid_positives.groupby('user_id', group_keys=False).sample(
            frac=test_ratio, random_state=random_state
        )
        
        # The rest of the positives go to the training set (matching by original index)
        train_positives = positives.drop(test_positives.index)
        
        # Recombine training set and reset indices cleanly
        train_df = pd.concat([train_positives, negatives]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        test_df = test_positives.reset_index(drop=True)
        
        print(f"Split Complete! Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df

    def _ndcg_at_k(self, recommended, ground_truth):
        dcg = 0.0
        idcg = 0.0
        
        for i, item in enumerate(recommended[:self.k]):
            if item in ground_truth:
                dcg += 1.0 / math.log2(i + 2)
                
        # Ideal DCG calculation
        ideal_hits = min(self.k, len(ground_truth))
        for i in range(ideal_hits):
            idcg += 1.0 / math.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0

    def _ap_at_k(self, recommended, ground_truth):
        hits = 0
        sum_precs = 0.0
        
        for i, item in enumerate(recommended[:self.k]):
            if item in ground_truth:
                hits += 1
                sum_precs += hits / (i + 1.0)
                
        return sum_precs / min(self.k, len(ground_truth)) if len(ground_truth) > 0 else 0.0

    def evaluate(self, model_name, recommendations_dict, test_df, total_catalog_items):
        """
        Calculates all 6 RecSys metrics for the given recommendations.
        
        :param model_name: String, name of the model (e.g., 'Memory-based Item-Item')
        :param recommendations_dict: Dictionary mapping user_id -> List of recommended anime_ids
        :param test_df: The test dataframe containing ground truth positive interactions
        :param total_catalog_items: Set or List of all unique anime_ids in the database
        :return: Dictionary containing the aggregated metrics
        """
        # Build ground truth dictionary mapping user_id -> set of true anime_ids
        ground_truth_dict = test_df.groupby('user_id')['anime_id'].apply(set).to_dict()
        
        precisions = []
        recalls = []
        f1_scores = []
        maps = []
        ndcgs = []
        
        recommended_unique_items = set()
        
        for user_id, true_items in ground_truth_dict.items():
            if user_id not in recommendations_dict:
                continue
                
            rec_items = recommendations_dict[user_id][:self.k]
            
            # Update items recommended globally for Catalog Coverage
            recommended_unique_items.update(rec_items)
            
            # Hit count
            hits = len(set(rec_items).intersection(true_items))
            
            # Precision & Recall
            precision = hits / self.k
            recall = hits / len(true_items)
            
            # F1-Score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            # Ranking Metrics
            maps.append(self._ap_at_k(rec_items, true_items))
            ndcgs.append(self._ndcg_at_k(rec_items, true_items))
            
        # Global Catalog Coverage
        catalog_coverage = len(recommended_unique_items) / len(total_catalog_items)
        
        # Aggregate Results
        results = {
            'Model': model_name,
            f'Precision@{self.k}': np.mean(precisions),
            f'Recall@{self.k}': np.mean(recalls),
            f'F1-Score@{self.k}': np.mean(f1_scores),
            f'MAP@{self.k}': np.mean(maps),
            f'NDCG@{self.k}': np.mean(ndcgs),
            'Catalog_Coverage': catalog_coverage
        }
        
        return results