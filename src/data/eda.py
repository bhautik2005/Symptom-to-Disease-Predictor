import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.helpers import setup_logger

logger = setup_logger("eda")

def run_eda(df: pd.DataFrame, config: dict):
    """Generates and saves exploratory data analysis plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned or raw dataset.
    config : dict
        Configuration dictionary.
    """
    plots_dir = config["data"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter symptom columns (exclude target labels)
    symptom_cols = [c for c in df.columns if c != 'diseases' and c != 'disease_encoded']
    disease_counts = df['diseases'].value_counts()
    
    # 1. Disease Distribution Plot
    logger.info("Generating disease distribution plot...")
    plt.figure(figsize=(16, 6))
    disease_counts.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title("Disease Distribution (all classes)", fontsize=14)
    plt.xlabel("Disease")
    plt.ylabel("Number of records")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "disease_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Top 20 symptoms plot
    logger.info("Generating top 20 symptoms plot...")
    symptom_freq = df[symptom_cols].sum().sort_values(ascending=False).head(20)
    plt.figure(figsize=(12, 5))
    symptom_freq.plot(kind='bar', color='coral', edgecolor='white')
    plt.title("Top 20 Most Frequent Symptoms", fontsize=13)
    plt.ylabel("Number of records with symptom")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top20_symptoms.png"), dpi=150)
    plt.close()
    
    # 3. Symptom-Disease association heatmap (top 20 symptoms vs top 15 diseases)
    logger.info("Generating symptom-disease association heatmap...")
    top15_diseases = disease_counts.head(15).index.tolist()
    top20_symptoms = symptom_freq.index.tolist()
    heatmap_data = (
        df[df['diseases'].isin(top15_diseases)]
        .groupby('diseases')[top20_symptoms]
        .mean()
    )
    plt.figure(figsize=(18, 7))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.3, annot=False)
    plt.title("Symptom–Disease Association Heatmap\n(mean symptom presence per disease)", fontsize=13)
    plt.xlabel("Symptoms")
    plt.ylabel("Diseases")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "symptom_disease_heatmap.png"), dpi=150)
    plt.close()
    
    # 4. Symptom co-occurrence matrix (top 30 symptoms)
    logger.info("Generating symptom co-occurrence matrix...")
    symptom_matrix = df[symptom_cols].astype(int)
    co_matrix = symptom_matrix.T.dot(symptom_matrix)
    np.fill_diagonal(co_matrix.values, 0)
    top30 = df[symptom_cols].sum().sort_values(ascending=False).head(30).index
    co_top = co_matrix.loc[top30, top30]
    
    plt.figure(figsize=(16, 13))
    sns.heatmap(co_top, cmap="Blues", linewidths=0.2, xticklabels=True, yticklabels=True, annot=False)
    plt.title("Symptom Co-occurrence Matrix (top 30 symptoms)", fontsize=13)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "symptom_cooccurrence.png"), dpi=150)
    plt.close()
    
    # Print top 10 most co-occurring pairs
    logger.info("Computing top 10 co-occurring symptom pairs...")
    upper = co_matrix.where(np.triu(np.ones(co_matrix.shape), k=1).astype(bool))
    top_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Symptom_A', 'level_1': 'Symptom_B', 0: 'co_count'})
        .sort_values('co_count', ascending=False)
        .head(10)
    )
    print("\n--- Top 10 Most Co-occurring Symptom Pairs ---")
    print(top_pairs.to_string(index=False))
    print()
    
    # 5. Signature symptom sets per disease
    logger.info("Generating signature symptoms per disease...")
    top10_diseases = disease_counts.head(10).index.tolist()
    
    print("\n--- Signature Symptoms per Disease ---")
    for disease in top10_diseases:
        in_disease  = df[df['diseases'] == disease][symptom_cols].mean()
        out_disease = df[df['diseases'] != disease][symptom_cols].mean()
        score = in_disease - out_disease
        top5 = score.sort_values(ascending=False).head(5)
        print(f"\n{disease}")
        for sym, val in top5.items():
            print(f"   {sym:<45} +{val:.2f}")
    print()
            
    # Visual: Top 5 signature symptoms for top 6 diseases
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, disease in enumerate(top10_diseases[:6]):
        in_d  = df[df['diseases'] == disease][symptom_cols].mean()
        out_d = df[df['diseases'] != disease][symptom_cols].mean()
        score = (in_d - out_d).sort_values(ascending=False).head(8)
        axes[i].barh(score.index[::-1], score.values[::-1], color='tomato')
        axes[i].set_title(disease, fontsize=10, fontweight='bold')
        axes[i].set_xlabel("Signature score", fontsize=8)
        axes[i].tick_params(axis='y', labelsize=7)
        axes[i].axvline(0, color='gray', linewidth=0.8)
    plt.suptitle("Signature Symptoms per Disease (top 6 diseases)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "signature_symptoms.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"EDA completed. Plots saved to: {plots_dir}")
