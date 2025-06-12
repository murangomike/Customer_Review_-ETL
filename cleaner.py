#!/usr/bin/env python3
"""
Data preprocessing and cleaning module
Performs NLP preprocessing, theme extraction, and model training
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import re
import joblib
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReviewProcessor:
    """Handles review data processing and ML model training"""
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        """
        Initialize the processor
        
        Args:
            n_topics: Number of topics for LDA
            random_state: Random state for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Model components
        self.vectorizer = None
        self.lda_model = None
        self.classifier = None
        self.topic_labels = {}
        
        logger.info("ReviewProcessor initialized")
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
        for data in required_data:
            try:
                nltk.download(data, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {data}: {e}")
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting dataframe cleaning")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['Unnamed: 0', 'Name']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Handle missing values
        df = df.dropna(subset=['review'])  # Remove rows without review text
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates(subset=['review'])
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Cleaning complete: {initial_shape} -> {df.shape}")
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for NLP analysis
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        # Remove stopwords and short words
        words = [word for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        # Stem words
        words = [self.stemmer.stem(word) for word in words]
        
        return " ".join(words)
    
    def extract_themes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract themes using LDA topic modeling
        
        Args:
            df: Dataframe with preprocessed text
            
        Returns:
            Dataframe with themes added
        """
        logger.info("Starting theme extraction")
        
        # Ensure we have preprocessed text
        if 'processed_review' not in df.columns:
            df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        # Remove empty processed reviews
        df = df[df['processed_review'].str.len() > 0]
        
        if len(df) == 0:
            logger.error("No valid reviews after preprocessing")
            return df
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=2,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        try:
            dtm = self.vectorizer.fit_transform(df['processed_review'])
        except ValueError as e:
            logger.error(f"Vectorization failed: {e}")
            return df
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=20
        )
        
        self.lda_model.fit(dtm)
        
        # Get topic assignments
        topic_values = self.lda_model.transform(dtm)
        df['topic'] = topic_values.argmax(axis=1)
        
        # Print top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        logger.info("Top words per topic:")
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            logger.info(f"Topic {idx}: {', '.join(top_words)}")
        
        # Define topic labels
        self.topic_labels = {
            0: 'Customer Support Issues',
            1: 'Billing & Subscription Problems',
            2: 'Service Quality Concerns',
            3: 'Account Management Issues',
            4: 'Delivery & Product Issues'
        }
        
        # Add theme labels
        df['theme'] = df['topic'].map(self.topic_labels)
        
        logger.info("Theme extraction completed")
        return df
    
    def train_classifier(self, df: pd.DataFrame) -> Dict:
        """
        Train a classification model
        
        Args:
            df: Dataframe with features and themes
            
        Returns:
            Dictionary with model metrics
        """
        logger.info("Starting model training")
        
        if 'theme' not in df.columns:
            logger.error("No themes found for training")
            return {}
        
        # Prepare features and target
        X = df['processed_review']
        y = df['theme']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Create pipeline
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])
        
        # Train model
        self.classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        logger.info(f"Model training completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return metrics
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.vectorizer:
            joblib.dump(self.vectorizer, f"{model_dir}/vectorizer.joblib")
        if self.lda_model:
            joblib.dump(self.lda_model, f"{model_dir}/lda_model.joblib")
        if self.classifier:
            joblib.dump(self.classifier, f"{model_dir}/classifier.joblib")
        
        logger.info(f"Models saved to {model_dir}")
    
    def process_pipeline(self, input_file: str, output_file: str) -> Dict:
        """
        Complete processing pipeline
        
        Args:
            input_file: Input CSV file
            output_file: Output CSV file
            
        Returns:
            Dictionary with processing metrics
        """
        logger.info("Starting complete processing pipeline")
        
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} reviews from {input_file}")
        
        # Clean dataframe
        df = self.clean_dataframe(df)
        
        # Preprocess text
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        # Extract themes
        df = self.extract_themes(df)
        
        # Train classifier
        metrics = self.train_classifier(df)
        
        # Save results
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        # Save models
        self.save_models()
        
        return metrics


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Process and analyze reviews')
    parser.add_argument('--input', '-i', default='raw_reviews.csv',
                       help='Input CSV file')
    parser.add_argument('--output', '-o', default='processed_reviews.csv',
                       help='Output CSV file')
    parser.add_argument('--topics', '-t', type=int, default=5,
                       help='Number of topics for LDA')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = ReviewProcessor(n_topics=args.topics)
        
        # Run processing pipeline
        metrics = processor.process_pipeline(args.input, args.output)
        
        # Log results
        if metrics:
            logger.info("Processing completed successfully")
            logger.info(f"Model Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"Model F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()