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
            }
        }

        stage('Extract') {
            steps {
                echo "📦 Extracting data from Neo4j"
                withCredentials([usernamePassword(credentialsId: 'neo4j-credentials', usernameVariable: 'NEO4J_USERNAME', passwordVariable: 'NEO4J_PASSWORD')]) {
                    sh '''
                        . ${VIRTUAL_ENV}/bin/activate
                        export NEO4J_URI="bolt://neo4j-server:7687"
                        export NEO4J_USERNAME="${NEO4J_USERNAME}"
                        export NEO4J_PASSWORD="${NEO4J_PASSWORD}"
                        python extract.py --output raw_reviews.csv
                        [ ! -f raw_reviews.csv ] && echo "❌ Extraction failed" && exit 1
                    '''
                }
                archiveArtifacts artifacts: 'raw_reviews.csv', allowEmptyArchive: false
            }
        }

        stage('Clean') {
            steps {
                echo "🧼 Cleaning and preprocessing data"
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    python cleaner.py --input raw_reviews.csv --output processed_reviews.csv --topics 5
                    [ ! -f processed_reviews.csv ] && echo "❌ Preprocessing failed" && exit 1
                    [ ! -d models ] && echo "❌ Model output directory missing" && exit 1
                '''
                archiveArtifacts artifacts: 'processed_reviews.csv,models/**', allowEmptyArchive: false
            }
        }

        stage('Load') {
            steps {
                echo "☁️ Uploading artifacts to S3"
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-credentials']]) {
                    sh '''
                        . ${VIRTUAL_ENV}/bin/activate
                        python loader.py --file processed_reviews.csv --bucket ${S3_BUCKET} --key data/processed_reviews_${BUILD_NUMBER}.csv
                        python loader.py --directory --file models --bucket ${S3_BUCKET} --key models/build_${BUILD_NUMBER}/
                    '''
                }
            }
        }

        stage('Analyze') {
            steps {
                echo "📊 Evaluating model performance"
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    python evaluate_model.py
                    [ ! -f model_metrics.json ] && echo "❌ Evaluation failed" && exit 1
                    cat model_metrics.json
                '''
                archiveArtifacts artifacts: 'model_metrics.json', allowEmptyArchive: false
            }
        }

        stage('Deploy Dashboard') {
            when { branch 'main' }
            steps {
                echo "🚀 Deploying dashboard"
                sh '''
                    . ${VIRTUAL_ENV}/bin/activate
                    timeout 30s streamlit run dashboard.py --server.headless true --server.port 8501 &
                    sleep 10
                    curl -f http://localhost:8501/_stcore/health || (echo "❌ Dashboard check failed" && pkill -f streamlit && exit 1)
                    pkill -f streamlit || true
                '''
            }
        }
    }

    post {
        always {
            echo "🧹 Cleaning up"
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
📊 Records: ${metrics.data_quality?.total_records ?: 'N/A'}
🎯 Themes: ${metrics.data_quality?.unique_themes ?: 'N/A'}
🔗 ${env.BUILD_URL}
                """
                emailext(
                    subject: "✅ Success - Build #${BUILD_NUMBER}",
                    body: message,
                    to: env.EMAIL_RECIPIENTS
                )
            }
        }

        failure {
            echo "❌ Pipeline failed!"
            emailext(
                subject: "❌ FAILED - Build #${BUILD_NUMBER}",
                body: "Check Jenkins logs\n${env.BUILD_URL}",
                to: env.EMAIL_RECIPIENTS
            )
        }
    }
}