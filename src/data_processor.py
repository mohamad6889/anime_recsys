import pandas as pd
import numpy as np
import random
from typing import Tuple

class AnimeDataPipeline:
    def __init__(self, ratings_path: str, anime_path: str, neg_ratio: int = 4, random_state: int = 42):
        print("Loading datasets...")
        self.ratings = pd.read_csv(ratings_path)
        self.animes = pd.read_csv(anime_path)
        self.neg_ratio = neg_ratio
        self.random_state = random_state

    def _clean_anime_metadata(self) -> pd.DataFrame:
        animes_clean = self.animes[['anime_id', 'genre']].copy()
        animes_clean['genre'] = animes_clean['genre'].fillna('Unknown')
        animes_clean['genre_list'] = animes_clean['genre'].apply(
            lambda x: [g.strip() for g in x.split(',')]
        )
        return animes_clean

    def _process_explicit_ratings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        explicit_df = df[df['rating'] != -1].copy()
        explicit_df['attracted'] = (explicit_df['rating'] >= 7).astype(int)
        
        exploded = explicit_df.explode('genre_list')
        user_genre_min = exploded.groupby(['user_id', 'genre_list'])['rating'].min().reset_index()
        user_genre_min.rename(columns={'rating': 'genre_avg_rating'}, inplace=True)
        
        return explicit_df, user_genre_min

    def _process_implicit_ratings(self, df: pd.DataFrame, user_genre_avg: pd.DataFrame) -> pd.DataFrame:
        implicit_df = df[df['rating'] == -1].copy()
        
        exploded_implicit = implicit_df.explode('genre_list')
        merged = exploded_implicit.merge(
            user_genre_avg, on=['user_id', 'genre_list'], how='left'
        )
        
        min_avg_scores = merged.groupby(['user_id', 'anime_id'])['genre_avg_rating'].mean().fillna(0).reset_index()
        min_avg_scores['attracted'] = (min_avg_scores['genre_avg_rating'] > 6).astype(int)
        
        implicit_clean = implicit_df.drop(columns=['rating', 'genre_list', 'genre']).merge(
            min_avg_scores[['user_id', 'anime_id', 'attracted']], 
            on=['user_id', 'anime_id'], 
            how='inner'
        )
        return implicit_clean

    def _generate_negative_samples(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        print("Generating negative samples...")
        all_anime_ids = set(self.animes['anime_id'].unique())
        
        user_interacted = processed_df.groupby('user_id')['anime_id'].apply(set).to_dict()
        user_positives = processed_df[processed_df['attracted'] == 1].groupby('user_id').size().to_dict()
        
        negative_samples_list = []
        
        # Lock the random generator for reproducibility
        rng = random.Random(self.random_state)
        
        for user, interacted_set in user_interacted.items():
            pos_count = user_positives.get(user, 0)
            if pos_count == 0:
                continue
                
            num_negatives_needed = pos_count * self.neg_ratio
            unseen_animes = list(all_anime_ids - interacted_set)
            
            num_to_sample = min(num_negatives_needed, len(unseen_animes))
            
            if num_to_sample > 0:
                # Use the locked generator instead of global random.sample
                sampled = rng.sample(unseen_animes, num_to_sample)
                negative_samples_list.extend([[user, a_id, 0] for a_id in sampled])
                
        neg_df = pd.DataFrame(negative_samples_list, columns=['user_id', 'anime_id', 'attracted'])
        return neg_df

    def execute(self) -> pd.DataFrame:
        animes_clean = self._clean_anime_metadata()
        merged_base = self.ratings.merge(animes_clean, on='anime_id', how='inner')
        
        print("Processing explicit ratings...")
        explicit_clean, user_genre_avg = self._process_explicit_ratings(merged_base)
        
        print("Processing implicit ratings...")
        implicit_clean = self._process_implicit_ratings(merged_base, user_genre_avg)
        
        explicit_clean = explicit_clean[['user_id', 'anime_id', 'attracted']]
        base_processed = pd.concat([explicit_clean, implicit_clean], ignore_index=True)
        
        negatives_df = self._generate_negative_samples(base_processed)
        
        # Shuffle with locked random_state
        final_dataset = pd.concat([base_processed, negatives_df], ignore_index=True)
        final_dataset = final_dataset.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Data pipeline complete. Total rows: {len(final_dataset)}")
        return final_dataset