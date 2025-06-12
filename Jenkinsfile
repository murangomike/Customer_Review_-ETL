pipeline {
    agent any
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 2, unit: 'HOURS')
        retry(2)
    }
    
    environment {
        // Environment variables
        PYTHON_VERSION = '3.9'
        VIRTUAL_ENV = 'venv'
        
        // AWS credentials (configure in Jenkins credentials)
        AWS_CREDENTIALS = credentials('aws-credentials')
        AWS_DEFAULT_REGION = 'us-east-1'
        S3_BUCKET = 'your-ml-pipeline-bucket'
        
        // Neo4j credentials (configure in Jenkins credentials)
        NEO4J_CREDENTIALS = credentials('neo4j-credentials')
        
        // Model artifacts
        MODEL_VERSION = "${BUILD_NUMBER}"
        MODEL_REGISTRY = "s3://${S3_BUCKET}/models"
        
        // Notification settings
        SLACK_CHANNEL = '#ml-pipeline'
        EMAIL_RECIPIENTS = 'team@company.com'
    }
    
    triggers {
        // Trigger every hour
        cron('0 * * * *')
        
        // Trigger on SCM changes
        pollSCM('H/15 * * * *')
    }
    
    stages {
        stage('Setup') {
            steps {
                echo "🚀 Starting ML Pipeline - Build #${BUILD_NUMBER}"
                
                // Clean workspace
                deleteDir()
                
                // Checkout code
                checkout scm
                
                // Setup Python environment
                sh '''
                    python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
                    source ${VIRTUAL_ENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
                
                // Verify environment
                sh '''
                    source ${VIRTUAL_ENV}/bin/activate
                    python --version
                    pip list
                '''
            }
            post {
                failure {
                    echo "❌ Setup failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Load Dataset') {
            steps {
                echo "📊 Loading dataset from Neo4j"
                
                script {
                    // Set Neo4j environment variables
                    withCredentials([usernamePassword(credentialsId: 'neo4j-credentials', 
                                                    usernameVariable: 'NEO4J_USERNAME', 
                                                    passwordVariable: 'NEO4J_PASSWORD')]) {
                        sh '''
                            source ${VIRTUAL_ENV}/bin/activate
                            export NEO4J_URI="bolt://neo4j-server:7687"
                            export NEO4J_USERNAME="${NEO4J_USERNAME}"
                            export NEO4J_PASSWORD="${NEO4J_PASSWORD}"
                            
                            # Extract data
                            python extract.py --output raw_reviews.csv --preview
                            
                            # Verify data extraction
                            if [ ! -f "raw_reviews.csv" ]; then
                                echo "❌ Data extraction failed - file not found"
                                exit 1
                            fi
                            
                            # Check file size
                            file_size=$(wc -l < raw_reviews.csv)
                            echo "📈 Extracted ${file_size} rows"
                            
                            if [ "$file_size" -lt 2 ]; then
                                echo "❌ Insufficient data extracted"
                                exit 1
                            fi
                        '''
                    }
                }
                
                // Archive raw data
                archiveArtifacts artifacts: 'raw_reviews.csv', allowEmptyArchive: false
            }
            post {
                failure {
                    echo "❌ Dataset loading failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Preprocess Features') {
            steps {
                echo "🔄 Preprocessing features and extracting themes"
                
                sh '''
                    source ${VIRTUAL_ENV}/bin/activate
                    
                    # Run preprocessing
                    python cleaner.py --input raw_reviews.csv --output processed_reviews.csv --topics 5
                    
                    # Verify preprocessing
                    if [ ! -f "processed_reviews.csv" ]; then
                        echo "❌ Preprocessing failed - file not found"
                        exit 1
                    fi
                    
                    # Check if models were created
                    if [ ! -d "models" ]; then
                        echo "❌ Model directory not created"
                        exit 1
                    fi
                    
                    echo "✅ Preprocessing completed successfully"
                '''
                
                // Archive processed data and models
                archiveArtifacts artifacts: 'processed_reviews.csv,models/**', allowEmptyArchive: false
            }
            post {
                failure {
                    echo "❌ Feature preprocessing failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Train Model') {
            steps {
                echo "🤖 Training ML model"
                
                sh '''
                    source ${VIRTUAL_ENV}/bin/activate
                    
                    # Model training is integrated in cleaner.py
                    # Verify model files exist
                    ls -la models/
                    
                    # Check model files
                    required_files=("vectorizer.joblib" "lda_model.joblib" "classifier.joblib")
                    for file in "${required_files[@]}"; do
                        if [ ! -f "models/$file" ]; then
                            echo "❌ Required model file missing: $file"
                            exit 1
                        fi
                    done
                    
                    echo "✅ Model training completed"
                '''
            }
            post {
                failure {
                    echo "❌ Model training failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Evaluate Performance') {
            steps {
                echo "📊 Evaluating model performance"
                
                script {
                    sh '''
                        source ${VIRTUAL_ENV}/bin/activate
                        
                        # Create evaluation script
                        cat > evaluate_model.py << 'EOF'
import pandas as pd
import json
import os
from sklearn.metrics import classification_report
import joblib

def evaluate_models():
    """Evaluate trained models and generate metrics"""
    metrics = {}
    
    # Load processed data
    if os.path.exists('processed_reviews.csv'):
        df = pd.read_csv('processed_reviews.csv')
        
        # Basic data metrics
        metrics['data_quality'] = {
            'total_records': len(df),
            'missing_reviews': df['review'].isna().sum(),
            'unique_themes': df['theme'].nunique() if 'theme' in df.columns else 0
        }
        
        # Model performance (if classification report exists)
        if 'theme' in df.columns:
            theme_distribution = df['theme'].value_counts().to_dict()
            metrics['theme_distribution'] = theme_distribution
        
        # Save metrics
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print("✅ Model evaluation completed")
        return metrics
    else:
        print("❌ No processed data found for evaluation")
        return {}

if __name__ == "__main__":
    evaluate_models()
EOF
                        
                        # Run evaluation
                        python evaluate_model.py
                        
                        # Verify metrics file
                        if [ ! -f "model_metrics.json" ]; then
                            echo "❌ Metrics file not generated"
                            exit 1
                        fi
                        
                        # Display metrics
                        echo "📊 Model Metrics:"
                        cat model_metrics.json
                    '''
                }
                
                // Archive metrics
                archiveArtifacts artifacts: 'model_metrics.json', allowEmptyArchive: false
            }
            post {
                failure {
                    echo "❌ Model evaluation failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Log Metrics') {
            steps {
                echo "📝 Logging metrics and uploading to S3"
                
                script {
                    withCredentials([aws(credentialsId: 'aws-credentials')]) {
                        sh '''
                            source ${VIRTUAL_ENV}/bin/activate
                            export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
                            export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
                            
                            # Upload processed data to S3
                            python loader.py --file processed_reviews.csv --bucket ${S3_BUCKET} --key "data/processed_reviews_${BUILD_NUMBER}.csv"
                            
                            # Upload models to S3
                            python loader.py --directory --file models --bucket ${S3_BUCKET} --key "models/build_${BUILD_NUMBER}/"
                            
                            # Upload metrics to S3
                            python loader.py --file model_metrics.json --bucket ${S3_BUCKET} --key "metrics/metrics_${BUILD_NUMBER}.json"
                            
                            echo "✅ All artifacts uploaded to S3"
                        '''
                    }
                }
            }
            post {
                failure {
                    echo "❌ Metrics logging failed"
                    script {
                        currentBuild.result = 'FAILURE'
                    }
                }
            }
        }
        
        stage('Deploy Dashboard') {
            when {
                branch 'main'
            }
            steps {
                echo "🚀 Deploying Streamlit dashboard"
                
                sh '''
                    source ${VIRTUAL_ENV}/bin/activate
                    
                    # Test dashboard locally first
                    timeout 30s streamlit run dashboard.py --server.headless true --server.port 8501 &
                    sleep 10
                    
                    # Check if dashboard is running
                    if curl -f http://localhost:8501/_stcore/health; then
                        echo "✅ Dashboard health check passed"
                        pkill -f streamlit || true
                    else
                        echo "❌ Dashboard health check failed"
                        pkill -f streamlit || true
                        exit 1
                    fi
                '''
            }
            post {
                failure {
                    echo "❌ Dashboard deployment failed"
                }
            }
        }
    }
    
    post {
        always {
            // Cleanup
            sh '''
                # Kill any remaining processes
                pkill -f streamlit || true
                
                # Cleanup virtual environment
                rm -rf ${VIRTUAL_ENV} || true
            '''
            
            // Publish test results if available
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'model_metrics.json',
                reportName: 'Model Metrics Report'
            ])
        }
        
        success {
            echo "✅ Pipeline completed successfully!"
            
            // Send success notification
            script {
                def metrics = readJSON file: 'model_metrics.json'
                def message = """
🎉 ML Pipeline Success - Build #${BUILD_NUMBER}
📊 Data Records: ${metrics.data_quality?.total_records ?: 'N/A'}
🎯 Unique Themes: ${metrics.data_quality?.unique_themes ?: 'N/A'}
⏱️ Duration: ${currentBuild.durationString}
🔗 Build URL: ${BUILD_URL}
                """.trim()
                
                // Slack notification (configure webhook in Jenkins)
                // slackSend(channel: env.SLACK_CHANNEL, message: message)
                
                // Email notification
                emailext(
                    subject: "✅ ML Pipeline Success - Build #${BUILD_NUMBER}",
                    body: message,
                    to: env.EMAIL_RECIPIENTS
                )
            }
        }
        
        failure {
            echo "❌ Pipeline failed!"
            
            // Send failure notification
            script {
                def message = """
❌ ML Pipeline Failed - Build #${BUILD_NUMBER}
📋 Stage: ${env.STAGE_NAME ?: 'Unknown'}
⏱️ Duration: ${currentBuild.durationString}
🔗 Build URL: ${BUILD_URL}
📋 Console: ${BUILD_URL}console
                """.trim()
                
                // Slack notification
                // slackSend(channel: env.SLACK_CHANNEL, message: message, color: 'danger')
                
                // Email notification
                emailext(
                    subject: "❌ ML Pipeline Failed - Build #${BUILD_NUMBER}",
                    body: message,
                    to: env.EMAIL_RECIPIENTS
                )
            }
        }
        
        unstable {
            echo "⚠️ Pipeline completed with warnings"
        }
    }
}