#!/usr/bin/env python3
"""
Data extraction module for Neo4j to Pandas DataFrame
Extracts review data from Neo4j graph database
"""

import os
import sys
import logging
import pandas as pd
from neo4j import GraphDatabase
from typing import Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jExtractor:
    """Handler for extracting data from Neo4j database"""
    
    def __init__(self):
        """Initialize Neo4j connection with environment variables"""
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME") 
        self.password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Missing required Neo4j environment variables")
            
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def fetch_all_reviews(self) -> pd.DataFrame:
        """
        Query all reviews from Neo4j database
        
        Returns:
            pd.DataFrame: DataFrame containing review data
        """
        query = """
        MATCH (r:RawReview)
        RETURN r.id AS id, 
               r.name AS name, 
               r.rating AS rating, 
               r.date AS date, 
               r.text AS review
        ORDER BY r.id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = result.data()
                
            if not records:
                logger.warning("No records found in database")
                return pd.DataFrame()
                
            df = pd.DataFrame(records)
            logger.info(f"Successfully extracted {len(df)} reviews")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch reviews: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Extract reviews from Neo4j')
    parser.add_argument('--output', '-o', default='raw_reviews.csv',
                       help='Output CSV file path')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Show preview of extracted data')
    
    args = parser.parse_args()
    
    extractor = None
    try:
        # Initialize extractor
        extractor = Neo4jExtractor()
        
        # Extract reviews
        df = extractor.fetch_all_reviews()
        
        if df.empty:
            logger.warning("No data extracted")
            sys.exit(1)
        
        # Save to CSV
        df.to_csv(args.output, index=False)
        logger.info(f"Data saved to {args.output}")
        
        # Show preview if requested
        if args.preview:
            print("\nData Preview:")
            print(f"Shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nData types:")
            print(df.dtypes)
            print(f"\nMissing values:\n{df.isnull().sum()}")
        
        logger.info("Data extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)
    finally:
        if extractor:
            extractor.close()


if __name__ == "__main__":
    main()