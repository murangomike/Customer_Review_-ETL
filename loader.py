#!/usr/bin/env python3
"""
S3 data loader module
Handles uploading processed data and models to S3 bucket
"""

import os
import sys
import logging
import boto3
import pandas as pd
import io
import json
import argparse
from datetime import datetime
from typing import Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class S3Loader:
    """Handles S3 upload operations"""
    
    def __init__(self, 
                 aws_access_key: Optional[str] = None,
                 aws_secret_key: Optional[str] = None,
                 region: str = 'us-east-1'):
        """
        Initialize S3 client
        
        Args:
            aws_access_key: AWS access key (optional, can use env vars)
            aws_secret_key: AWS secret key (optional, can use env vars)
            region: AWS region
        """
        self.aws_access_key = aws_access_key or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = region
        
        self.s3_client = self._initialize_s3_client()
        
    def _initialize_s3_client(self):
        """Initialize S3 client with credentials"""
        try:
            if self.aws_access_key and self.aws_secret_key:
                session = boto3.Session(
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.region
                )
                s3_client = session.client('s3')
            else:
                # Use default credentials (environment, IAM role, etc.)
                s3_client = boto3.client('s3', region_name=self.region)
            
            # Test connection
            s3_client.list_buckets()
            logger.info("Successfully initialized S3 client")
            return s3_client
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise
    
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if S3 bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                return False
            else:
                logger.error(f"Error checking bucket {bucket_name}: {e}")
                raise
    
    def upload_dataframe(self, 
                        df: pd.DataFrame, 
                        bucket_name: str, 
                        object_key: str,
                        file_format: str = 'csv') -> bool:
        """
        Upload pandas DataFrame to S3
        
        Args:
            df: DataFrame to upload
            bucket_name: S3 bucket name
            object_key: S3 object key (path)
            file_format: File format ('csv' or 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.bucket_exists(bucket_name):
                logger.error(f"Bucket {bucket_name} does not exist")
                return False
            
            # Prepare file buffer
            if file_format.lower() == 'csv':
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                content = buffer.getvalue()
                content_type = 'text/csv'
            elif file_format.lower() == 'parquet':
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                content = buffer.getvalue()
                content_type = 'application/octet-stream'
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return False
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=content,
                ContentType=content_type
            )
            
            logger.info(f"✅ Successfully uploaded DataFrame to s3://{bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {e}")
            return False
    
    def upload_file(self, 
                   file_path: str, 
                   bucket_name: str, 
                   object_key: str) -> bool:
        """
        Upload local file to S3
        
        Args:
            file_path: Local file path
            bucket_name: S3 bucket name
            object_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File {file_path} does not exist")
                return False
            
            if not self.bucket_exists(bucket_name):
                logger.error(f"Bucket {bucket_name} does not exist")
                return False
            
            self.s3_client.upload_file(file_path, bucket_name, object_key)
            logger.info(f"✅ Successfully uploaded {file_path} to s3://{bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {e}")
            return False
    
    def upload_directory(self, 
                        directory_path: str, 
                        bucket_name: str, 
                        prefix: str = "") -> bool:
        """
        Upload entire directory to S3
        
        Args:
            directory_path: Local directory path
            bucket_name: S3 bucket name
            prefix: S3 prefix (folder path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(directory_path):
                logger.error(f"Directory {directory_path} does not exist")
                return False
            
            success_count = 0
            error_count = 0
            
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, directory_path)
                    s3_key = os.path.join(prefix, relative_path).replace('\\', '/')
                    
                    if self.upload_file(local_path, bucket_name, s3_key):
                        success_count += 1
                    else:
                        error_count += 1
            
            logger.info(f"Directory upload completed: {success_count} files uploaded, {error_count} errors")
            return error_count == 0
            
        except Exception as e:
            logger.error(f"Failed to upload directory {directory_path}: {e}")
            return False
    
    def upload_metrics(self, 
                      metrics: Dict, 
                      bucket_name: str, 
                      object_key: str) -> bool:
        """
        Upload metrics dictionary as JSON to S3
        
        Args:
            metrics: Metrics dictionary
            bucket_name: S3 bucket name
            object_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Convert to JSON
            json_content = json.dumps(metrics, indent=2, default=str)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json_content,
                ContentType='application/json'
            )
            
            logger.info(f"✅ Successfully uploaded metrics to s3://{bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload metrics: {e}")
            return False
    
    def download_file(self, 
                     bucket_name: str, 
                     object_key: str, 
                     local_path: str) -> bool:
        """
        Download file from S3
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key
            local_path: Local file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.download_file(bucket_name, object_key, local_path)
            logger.info(f"✅ Successfully downloaded s3://{bucket_name}/{object_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def load_dataframe_from_s3(self, 
                              bucket_name: str, 
                              object_key: str,
                              file_format: str = 'csv') -> Optional[pd.DataFrame]:
        """
        Load DataFrame from S3
        
        Args:
            bucket_name: S3 bucket name
            object_key: S3 object key
            file_format: File format ('csv' or 'parquet')
            
        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            
            if file_format.lower() == 'csv':
                df = pd.read_csv(io.BytesIO(response['Body'].read()))
            elif file_format.lower() == 'parquet':
                df = pd.read_parquet(io.BytesIO(response['Body'].read()))
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return None
            
            logger.info(f"✅ Successfully loaded DataFrame from s3://{bucket_name}/{object_key}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame from S3: {e}")
            return None


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Upload data to S3')
    parser.add_argument('--file', '-f', required=True,
                       help='File to upload')
    parser.add_argument('--bucket', '-b', required=True,
                       help='S3 bucket name')
    parser.add_argument('--key', '-k', required=True,
                       help='S3 object key')
    parser.add_argument('--format', default='csv', choices=['csv', 'parquet'],
                       help='File format for DataFrames')
    parser.add_argument('--directory', '-d', action='store_true',
                       help='Upload entire directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize S3 loader
        loader = S3Loader()
        
        if args.directory:
            # Upload directory
            success = loader.upload_directory(args.file, args.bucket, args.key)
        else:
            # Check if file is a CSV (DataFrame) or regular file
            if args.file.endswith('.csv') and not args.directory:
                # Load and upload as DataFrame
                df = pd.read_csv(args.file)
                success = loader.upload_dataframe(df, args.bucket, args.key, args.format)
            else:
                # Upload as regular file
                success = loader.upload_file(args.file, args.bucket, args.key)
        
        if success:
            logger.info("Upload completed successfully")
        else:
            logger.error("Upload failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Upload operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()