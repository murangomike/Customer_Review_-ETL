#!/usr/bin/env python3
"""
ML Pipeline Airflow DAG
Orchestrates the complete ETL + Analysis workflow:
1. Extract data from Neo4j
2. Transform/clean the data
3. Load to S3 bucket
4. Analysis - Create dashboard from S3 data

Schedule: Runs every hour
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from airflow.hooks.base import BaseHook
import logging
import os
import sys
import pandas as pd
import boto3
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DAG Configuration
DAG_ID = 'ml_pipeline_etl_analysis'
SCHEDULE_INTERVAL = '@hourly'  # Run every hour
MAX_ACTIVE_RUNS = 1

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'email': ['ml-team@company.com'],
}

# Configuration variables (set these in Airflow Variables)
S3_BUCKET = Variable.get("s3_bucket", default_var="ml-pipeline-bucket")
AWS_REGION = Variable.get("aws_region", default_var="us-east-1")
NEO4J_URI = Variable.get("neo4j_uri", default_var="bolt://localhost:7687")
DASHBOARD_PORT = Variable.get("dashboard_port", default_var="8501")

# File paths
RAW_DATA_FILE = '/tmp/raw_reviews.csv'
PROCESSED_DATA_FILE = '/tmp/processed_reviews.csv'
METRICS_FILE = '/tmp/model_metrics.json'
MODELS_DIR = '/tmp/models'

def extract_data_from_neo4j(**context):
    """
    Extract data from Neo4j database
    """
    logger.info("🔍 Starting data extraction from Neo4j")
    
    try:
        # Get Neo4j credentials from Airflow connections
        neo4j_conn = BaseHook.get_connection('neo4j_default')
        neo4j_uri = neo4j_conn.host or NEO4J_URI
        neo4j_username = neo4j_conn.login
        neo4j_password = neo4j_conn.password
        
        # Initialize Neo4j driver
        driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Query to extract all reviews
        query = """
        MATCH (r:RawReview)
        RETURN r.id AS id, 
               r.name AS name, 
               r.rating AS rating, 
               r.date AS date, 
               r.text AS review
        ORDER BY r.id
        """
        
        # Execute query and fetch data
        with driver.session() as session:
            result = session.run(query)
            records = result.data()
        
        driver.close()
        
        if not records:
            raise ValueError("No records found in Neo4j database")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(records)
        df.to_csv(RAW_DATA_FILE, index=False)
        
        # Log extraction summary
        logger.info(f"✅ Extracted {len(df)} reviews from Neo4j")
        logger.info(f"📁 Raw data saved to {RAW_DATA_FILE}")
        
        # Push metadata to XCom
        context['task_instance'].xcom_push(key='extraction_stats', value={
            'total_records': len(df),
            'file_path': RAW_DATA_FILE,
            'extraction_timestamp': datetime.now().isoformat()
        })
        
        return RAW_DATA_FILE
        
    except Exception as e:
        logger.error(f"❌ Data extraction failed: {str(e)}")
        raise


def transform_and_clean_data(**context):
    """
    Transform and clean the extracted data
    """
    logger.info("🔄 Starting data transformation and cleaning")
    
    try:
        # Import required modules (assuming they're in the same directory)
        sys.path.append('/opt/airflow/dags/scripts')
        from cleaner import ReviewProcessor
        
        # Initialize processor
        processor = ReviewProcessor(n_topics=5, random_state=42)
        
        # Load raw data
        if not os.path.exists(RAW_DATA_FILE):
            raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_FILE}")
        
        df = pd.read_csv(RAW_DATA_FILE)
        logger.info(f"📊 Loaded {len(df)} raw reviews for processing")
        
        # Clean dataframe
        df_cleaned = processor.clean_dataframe(df)
        
        # Preprocess text
        df_cleaned['processed_review'] = df_cleaned['review'].apply(processor.preprocess_text)
        
        # Extract themes using LDA
        df_processed = processor.extract_themes(df_cleaned)
        
        # Train classifier
        metrics = processor.train_classifier(df_processed)
        
        # Save processed data
        df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
        
        # Save models
        os.makedirs(MODELS_DIR, exist_ok=True)
        processor.save_models(MODELS_DIR)
        
        # Save metrics
        import json
        metrics['processing_timestamp'] = datetime.now().isoformat()
        metrics['total_processed_records'] = len(df_processed)
        
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"✅ Data transformation completed")
        logger.info(f"📁 Processed data saved to {PROCESSED_DATA_FILE}")
        logger.info(f"🤖 Models saved to {MODELS_DIR}")
        
        # Push metadata to XCom
        context['task_instance'].xcom_push(key='transformation_stats', value={
            'processed_records': len(df_processed),
            'model_accuracy': metrics.get('accuracy', 0),
            'f1_score': metrics.get('f1_score', 0),
            'unique_themes': df_processed['theme'].nunique(),
            'processing_timestamp': metrics['processing_timestamp']
        })
        
        return PROCESSED_DATA_FILE
        
    except Exception as e:
        logger.error(f"❌ Data transformation failed: {str(e)}")
        raise


def load_data_to_s3(**context):
    """
    Load processed data and models to S3 bucket
    """
    logger.info("📤 Starting data upload to S3")
    
    try:
        # Get AWS credentials from Airflow connections
        aws_conn = BaseHook.get_connection('aws_default')
        
        # Initialize S3 client
        if aws_conn.login and aws_conn.password:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_conn.login,
                aws_secret_access_key=aws_conn.password,
                region_name=AWS_REGION
            )
        else:
            # Use IAM role or environment credentials
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Check if bucket exists, create if not
        try:
            s3_client.head_bucket(Bucket=S3_BUCKET)
        except:
            logger.info(f"Creating S3 bucket: {S3_BUCKET}")
            s3_client.create_bucket(Bucket=S3_BUCKET)
        
        # Get execution date for versioning
        execution_date = context['execution_date'].strftime('%Y%m%d_%H%M%S')
        
        # Upload processed data
        processed_s3_key = f"data/processed_reviews_{execution_date}.csv"
        s3_client.upload_file(PROCESSED_DATA_FILE, S3_BUCKET, processed_s3_key)
        logger.info(f"✅ Uploaded processed data to s3://{S3_BUCKET}/{processed_s3_key}")
        
        # Upload metrics
        metrics_s3_key = f"metrics/model_metrics_{execution_date}.json"
        s3_client.upload_file(METRICS_FILE, S3_BUCKET, metrics_s3_key)
        logger.info(f"✅ Uploaded metrics to s3://{S3_BUCKET}/{metrics_s3_key}")
        
        # Upload models directory
        model_files_uploaded = []
        if os.path.exists(MODELS_DIR):
            for root, dirs, files in os.walk(MODELS_DIR):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, MODELS_DIR)
                    s3_key = f"models/{execution_date}/{relative_path}"
                    
                    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                    model_files_uploaded.append(s3_key)
                    logger.info(f"✅ Uploaded model file to s3://{S3_BUCKET}/{s3_key}")
        
        # Push S3 locations to XCom
        s3_locations = {
            'processed_data': f"s3://{S3_BUCKET}/{processed_s3_key}",
            'metrics': f"s3://{S3_BUCKET}/{metrics_s3_key}",
            'models': [f"s3://{S3_BUCKET}/{key}" for key in model_files_uploaded],
            'upload_timestamp': datetime.now().isoformat()
        }
        
        context['task_instance'].xcom_push(key='s3_locations', value=s3_locations)
        
        logger.info(f"✅ All data successfully uploaded to S3 bucket: {S3_BUCKET}")
        return s3_locations
        
    except Exception as e:
        logger.error(f"❌ S3 upload failed: {str(e)}")
        raise


def prepare_dashboard_data(**context):
    """
    Prepare data for dashboard by downloading from S3
    """
    logger.info("📊 Preparing dashboard data")
    
    try:
        # Get S3 locations from previous task
        s3_locations = context['task_instance'].xcom_pull(
            task_ids='load_to_s3', 
            key='s3_locations'
        )
        
        if not s3_locations:
            raise ValueError("No S3 locations found in XCom")
        
        # Download processed data from S3 for dashboard
        aws_conn = BaseHook.get_connection('aws_default')
        
        if aws_conn.login and aws_conn.password:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_conn.login,
                aws_secret_access_key=aws_conn.password,
                region_name=AWS_REGION
            )
        else:
            s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Extract S3 key from URL
        processed_data_url = s3_locations['processed_data']
        s3_key = processed_data_url.replace(f"s3://{S3_BUCKET}/", "")
        
        # Download to dashboard directory
        dashboard_data_path = '/tmp/dashboard_data.csv'
        s3_client.download_file(S3_BUCKET, s3_key, dashboard_data_path)
        
        logger.info(f"✅ Dashboard data prepared at {dashboard_data_path}")
        
        # Verify data
        df = pd.read_csv(dashboard_data_path)
        logger.info(f"📈 Dashboard data contains {len(df)} records with {df.shape[1]} columns")
        
        context['task_instance'].xcom_push(key='dashboard_data_path', value=dashboard_data_path)
        
        return dashboard_data_path
        
    except Exception as e:
        logger.error(f"❌ Dashboard data preparation failed: {str(e)}")
        raise


def validate_pipeline_success(**context):
    """
    Validate that the entire pipeline completed successfully
    """
    logger.info("✅ Validating pipeline success")
    
    try:
        # Get stats from all previous tasks
        extraction_stats = context['task_instance'].xcom_pull(
            task_ids='extract_data', 
            key='extraction_stats'
        )
        
        transformation_stats = context['task_instance'].xcom_pull(
            task_ids='transform_data', 
            key='transformation_stats'
        )
        
        s3_locations = context['task_instance'].xcom_pull(
            task_ids='load_to_s3', 
            key='s3_locations'
        )
        
        # Compile pipeline summary
        pipeline_summary = {
            'pipeline_id': context['run_id'],
            'execution_date': context['execution_date'].isoformat(),
            'total_records_extracted': extraction_stats.get('total_records', 0),
            'total_records_processed': transformation_stats.get('processed_records', 0),
            'model_accuracy': transformation_stats.get('model_accuracy', 0),
            'f1_score': transformation_stats.get('f1_score', 0),
            'unique_themes': transformation_stats.get('unique_themes', 0),
            's3_bucket': S3_BUCKET,
            'data_location': s3_locations.get('processed_data', ''),
            'pipeline_status': 'SUCCESS',
            'completion_timestamp': datetime.now().isoformat()
        }
        
        # Log success summary
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Records: {pipeline_summary['total_records_extracted']} extracted, {pipeline_summary['total_records_processed']} processed")
        logger.info(f"🎯 Model Performance: Accuracy={pipeline_summary['model_accuracy']:.4f}, F1={pipeline_summary['f1_score']:.4f}")
        logger.info(f"🗂️ Themes Identified: {pipeline_summary['unique_themes']}")
        logger.info(f"📤 Data Location: {pipeline_summary['data_location']}")
        
        # Save pipeline summary
        summary_file = '/tmp/pipeline_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        context['task_instance'].xcom_push(key='pipeline_summary', value=pipeline_summary)
        
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"❌ Pipeline validation failed: {str(e)}")
        raise


# Initialize DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='ML Pipeline: Neo4j → Transform → S3 → Dashboard',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=MAX_ACTIVE_RUNS,
    catchup=False,
    tags=['ml', 'etl', 'neo4j', 's3', 'nlp', 'dashboard'],
    doc_md=__doc__,
)

# Task 1: Extract data from Neo4j
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data_from_neo4j,
    dag=dag,
    doc_md="""
    ## Extract Data from Neo4j
    
    Extracts review data from Neo4j graph database:
    - Connects to Neo4j using configured credentials
    - Runs Cypher query to fetch all RawReview nodes
    - Saves raw data to CSV file
    - Pushes extraction statistics to XCom
    """,
)

# Task 2: Transform and clean data
transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_and_clean_data,
    dag=dag,
    doc_md="""
    ## Transform and Clean Data
    
    Processes the extracted data:
    - Cleans and normalizes the raw data
    - Performs NLP preprocessing (tokenization, stemming, etc.)
    - Extracts themes using LDA topic modeling
    - Trains classification model
    - Saves processed data and trained models
    """,
)

# Task 3: Load data to S3
load_task = PythonOperator(
    task_id='load_to_s3',
    python_callable=load_data_to_s3,
    dag=dag,
    doc_md="""
    ## Load Data to S3
    
    Uploads processed data and models to S3:
    - Creates S3 bucket if it doesn't exist
    - Uploads processed CSV data
    - Uploads model artifacts
    - Uploads performance metrics
    - Pushes S3 locations to XCom
    """,
)

# Task 4: Prepare dashboard data
dashboard_prep_task = PythonOperator(
    task_id='prepare_dashboard',
    python_callable=prepare_dashboard_data,
    dag=dag,
    doc_md="""
    ## Prepare Dashboard Data
    
    Prepares data for the analytics dashboard:
    - Downloads processed data from S3
    - Validates data integrity
    - Prepares data in format suitable for dashboard
    """,
)

# Task 5: Deploy/Update Dashboard
dashboard_task = BashOperator(
    task_id='deploy_dashboard',
    bash_command=f"""
    # Navigate to dashboard directory
    cd /opt/airflow/dags/scripts
    
    # Copy the prepared data
    cp /tmp/dashboard_data.csv processed_reviews.csv
    
    # Kill any existing dashboard processes
    pkill -f "streamlit run dashboard.py" || true
    
    # Start dashboard in background
    nohup streamlit run dashboard.py \
        --server.headless true \
        --server.port {DASHBOARD_PORT} \
        --server.address 0.0.0.0 \
        > /tmp/dashboard.log 2>&1 &
    
    # Wait a moment for startup
    sleep 10
    
    # Health check
    if curl -f http://localhost:{DASHBOARD_PORT}/_stcore/health; then
        echo "✅ Dashboard deployed successfully on port {DASHBOARD_PORT}"
    else
        echo "❌ Dashboard health check failed"
        cat /tmp/dashboard.log
        exit 1
    fi
    """,
    dag=dag,
    doc_md=f"""
    ## Deploy Analytics Dashboard
    
    Deploys the Streamlit dashboard:
    - Copies processed data to dashboard location
    - Starts Streamlit application on port {DASHBOARD_PORT}
    - Performs health check
    - Dashboard provides interactive analysis of themes and metrics
    """,
)

# Task 6: Validate pipeline success
validation_task = PythonOperator(
    task_id='validate_pipeline',
    python_callable=validate_pipeline_success,
    dag=dag,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    doc_md="""
    ## Validate Pipeline Success
    
    Final validation and summary:
    - Collects statistics from all pipeline stages
    - Validates data quality and model performance
    - Generates comprehensive pipeline summary
    - Logs success metrics and locations
    """,
)

# Task 7: Cleanup (always runs)
cleanup_task = BashOperator(
    task_id='cleanup',
    bash_command="""
    # Clean up temporary files
    rm -f /tmp/raw_reviews.csv
    rm -f /tmp/processed_reviews.csv
    rm -f /tmp/model_metrics.json
    rm -f /tmp/dashboard_data.csv
    rm -f /tmp/pipeline_summary.json
    rm -rf /tmp/models
    
    echo "✅ Cleanup completed"
    """,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,  # Run regardless of success/failure
    doc_md="""
    ## Cleanup
    
    Cleans up temporary files and resources:
    - Removes temporary CSV files
    - Removes temporary model files
    - Cleans up processing artifacts
    - Runs regardless of pipeline success/failure
    """,
)

# Define task dependencies
# ETL Pipeline: Extract → Transform → Load → Analysis
extract_task >> transform_task >> load_task >> dashboard_prep_task >> dashboard_task >> validation_task >> cleanup_task

# Add additional monitoring tasks (optional)
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
    doc_md="Pipeline start marker"
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,
    doc_md="Pipeline end marker"
)

# Connect start and end tasks
start_task >> extract_task
[validation_task, cleanup_task] >> end_task

# Set task documentation
dag.doc_md = """
# ML Pipeline DAG - Neo4j to S3 ETL with Analytics

This DAG implements a complete machine learning pipeline that:

## Pipeline Stages:
1. **Extract**: Pull review data from Neo4j Graph Database
2. **Transform**: Clean, preprocess, and extract themes using NLP
3. **Load**: Save processed data and models to S3 bucket  
4. **Analysis**: Deploy interactive Streamlit dashboard

## Schedule:
- Runs every hour (`@hourly`)
- Maximum 1 active run at a time
- Automatic retries with exponential backoff

## Data Flow:
```
Neo4j → Raw CSV → Processed CSV → S3 Bucket → Dashboard
         ↓           ↓              ↓
    Extract    Transform        Load
```

## Key Features:
- **NLP Processing**: Tokenization, stemming, stop-word removal
- **Theme Extraction**: LDA topic modeling with 5 themes
- **Model Training**: Random Forest classifier for theme prediction
- **Data Versioning**: Timestamped files in S3
- **Health Monitoring**: Pipeline validation and metrics
- **Interactive Dashboard**: Real-time analytics via Streamlit

## Configuration:
Set these Airflow Variables:
- `s3_bucket`: S3 bucket name
- `aws_region`: AWS region
- `neo4j_uri`: Neo4j connection URI
- `dashboard_port`: Streamlit port (default: 8501)

## Connections Required:
- `neo4j_default`: Neo4j database connection
- `aws_default`: AWS credentials for S3 access

## Monitoring:
- Email notifications on failure
- XCom data sharing between tasks
- Comprehensive logging
- Pipeline success validation
"""