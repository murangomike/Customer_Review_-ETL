pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 2, unit: 'HOURS')
        retry(2)
    }

    environment {
        PYTHON_VERSION = '3.9'
        VIRTUAL_ENV = 'venv'

        AWS_DEFAULT_REGION = 'us-east-1'
        S3_BUCKET = 'your-ml-pipeline-bucket'

        MODEL_VERSION = "${BUILD_NUMBER}"
        MODEL_REGISTRY = "s3://${S3_BUCKET}/models"

        SLACK_CHANNEL = '#ml-pipeline'
        EMAIL_RECIPIENTS = 'mikemurango00@gmail.com'
    }

    triggers {
        cron('0 * * * *')
        pollSCM('H/15 * * * *')
    }

    stages {
        stage('Setup') {
            steps {
                echo "🚀 Starting ML Pipeline - Build #${BUILD_NUMBER}"
                deleteDir()
                checkout scm

                sh '''
                    python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
                    . ${VIRTUAL_ENV}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''

                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    python --version
                    pip list
                '''
            }
        }

        stage('Load Dataset') {
            steps {
                echo "📊 Loading dataset from Neo4j"

                withCredentials([usernamePassword(credentialsId: 'neo4j-credentials', usernameVariable: 'NEO4J_USERNAME', passwordVariable: 'NEO4J_PASSWORD')]) {
                    sh '''
                        . ${VIRTUAL_ENV}/bin/activate
                        export NEO4J_URI="bolt://neo4j-server:7687"
                        export NEO4J_USERNAME="${NEO4J_USERNAME}"
                        export NEO4J_PASSWORD="${NEO4J_PASSWORD}"

                        python extract.py --output raw_reviews.csv --preview

                        if [ ! -f "raw_reviews.csv" ]; then
                            echo "❌ Data extraction failed - file not found"
                            exit 1
                        fi

                        file_size=$(wc -l < raw_reviews.csv)
                        echo "📈 Extracted ${file_size} rows"

                        if [ "$file_size" -lt 2 ]; then
                            echo "❌ Insufficient data extracted"
                            exit 1
                        fi
                    '''
                }

                archiveArtifacts artifacts: 'raw_reviews.csv', allowEmptyArchive: false
            }
        }

        stage('Preprocess Features') {
            steps {
                echo "🔄 Preprocessing features and extracting themes"

                sh '''
                    . ${VIRTUAL_ENV}/bin/activate

                    python cleaner.py --input raw_reviews.csv --output processed_reviews.csv --topics 5

                    if [ ! -f "processed_reviews.csv" ]; then
                        echo "❌ Preprocessing failed - file not found"
                        exit 1
                    fi

                    if [ ! -d "models" ]; then
                        echo "❌ Model directory not created"
                        exit 1
                    fi
                '''

                archiveArtifacts artifacts: 'processed_reviews.csv,models/**', allowEmptyArchive: false
            }
        }

        stage('Train Model') {
            steps {
                echo "🤖 Training ML model"

                sh '''
                    . ${VIRTUAL_ENV}/bin/activate

                    ls -la models/

                    for file in vectorizer.joblib lda_model.joblib classifier.joblib; do
                        if [ ! -f "models/$file" ]; then
                            echo "❌ Required model file missing: $file"
                            exit 1
                        fi
                    done
                '''
            }
        }

        stage('Evaluate Performance') {
            steps {
                echo "📊 Evaluating model performance"

                sh '''
                    . ${VIRTUAL_ENV}/bin/activate

                    cat > evaluate_model.py << 'EOF'
import pandas as pd
import json
import os

def evaluate_models():
    metrics = {}
    if os.path.exists('processed_reviews.csv'):
        df = pd.read_csv('processed_reviews.csv')
        metrics['data_quality'] = {
            'total_records': len(df),
            'missing_reviews': df['review'].isna().sum(),
            'unique_themes': df['theme'].nunique() if 'theme' in df.columns else 0
        }
        if 'theme' in df.columns:
            metrics['theme_distribution'] = df['theme'].value_counts().to_dict()
        with open('model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✅ Model evaluation completed")
    else:
        print("❌ No processed data found for evaluation")

if __name__ == "__main__":
    evaluate_models()
EOF

                    python evaluate_model.py

                    if [ ! -f "model_metrics.json" ]; then
                        echo "❌ Metrics file not generated"
                        exit 1
                    fi

                    echo "📊 Model Metrics:"
                    cat model_metrics.json
                '''

                archiveArtifacts artifacts: 'model_metrics.json', allowEmptyArchive: false
            }
        }

        stage('Log Metrics') {
            steps {
                echo "📝 Logging metrics and uploading to S3"

                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-credentials']]) {
                    sh '''
                        . ${VIRTUAL_ENV}/bin/activate

                        export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
                        export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

                        python loader.py --file processed_reviews.csv --bucket ${S3_BUCKET} --key "data/processed_reviews_${BUILD_NUMBER}.csv"
                        python loader.py --directory --file models --bucket ${S3_BUCKET} --key "models/build_${BUILD_NUMBER}/"
                        python loader.py --file model_metrics.json --bucket ${S3_BUCKET} --key "metrics/metrics_${BUILD_NUMBER}.json"
                    '''
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
                    . ${VIRTUAL_ENV}/bin/activate

                    timeout 30s streamlit run dashboard.py --server.headless true --server.port 8501 &
                    sleep 10

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
        }
    }

    post {
        always {
            sh '''
                pkill -f streamlit || true
                rm -rf ${VIRTUAL_ENV} || true
            '''

            publishHTML([
                reportDir: '.',
                reportFiles: 'model_metrics.json',
                reportName: 'Model Metrics Report',
                keepAll: true
            ])
        }

        success {
            echo "✅ Pipeline completed successfully!"

            script {
                def metrics = readJSON file: 'model_metrics.json'
                def message = """
🎉 ML Pipeline Success - Build #${BUILD_NUMBER}
📊 Data Records: ${metrics.data_quality?.total_records ?: 'N/A'}
🎯 Unique Themes: ${metrics.data_quality?.unique_themes ?: 'N/A'}
⏱️ Duration: ${currentBuild.durationString}
🔗 Build URL: ${env.BUILD_URL}
                """.trim()

                // slackSend(channel: env.SLACK_CHANNEL, message: message) // Optional
                emailext(
                    subject: "✅ ML Pipeline Success - Build #${BUILD_NUMBER}",
                    body: message,
                    to: env.EMAIL_RECIPIENTS
                )
            }
        }

        failure {
            echo "❌ Pipeline failed!"
            script {
                emailext(
                    subject: "❌ ML Pipeline FAILED - Build #${BUILD_NUMBER}",
                    body: "Check the Jenkins logs for more details.\nBuild URL: ${env.BUILD_URL}",
                    to: env.EMAIL_RECIPIENTS
                )
            }
        }
    }
}