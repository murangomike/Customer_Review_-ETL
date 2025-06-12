# Multi-stage Dockerfile for ML Pipeline
# Supports both Airflow DAG and Jenkins pipeline execution

# Stage 1: Base Python environment with common dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic system tools
    curl \
    wget \
    git \
    unzip \
    # Build tools for Python packages
    build-essential \
    gcc \
    g++ \
    # Java for Airflow and Spark compatibility
    openjdk-11-jre-headless \
    # Network tools
    netcat-traditional \
    # Process management
    supervisor \
    # Text processing
    vim \
    nano \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Airflow-specific setup
FROM base as airflow-base

# Install Airflow with specific providers
RUN pip install --no-cache-dir \
    apache-airflow==2.7.0 \
    apache-airflow-providers-amazon==8.6.0 \
    apache-airflow-providers-neo4j==3.3.0 \
    apache-airflow-providers-postgres==5.6.0

# Create airflow user and directories
RUN useradd --create-home --shell /bin/bash airflow && \
    mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config

# Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
ENV AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here

# Stage 3: Jenkins-compatible setup
FROM base as jenkins-base

# Install Jenkins dependencies
RUN apt-get update && apt-get install -y \
    openssh-client \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create jenkins user
RUN useradd --create-home --shell /bin/bash jenkins

# Stage 4: Final production image
FROM base as production

# Copy Airflow installation from airflow-base
COPY --from=airflow-base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=airflow-base /usr/local/bin /usr/local/bin

# Create users and directories
RUN useradd --create-home --shell /bin/bash airflow && \
    useradd --create-home --shell /bin/bash jenkins && \
    mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/config && \
    mkdir -p /opt/jenkins && \
    mkdir -p /app/scripts /app/models /app/data

# Set proper permissions
RUN chown -R airflow:airflow /opt/airflow && \
    chown -R jenkins:jenkins /opt/jenkins && \
    chown -R airflow:airflow /app

# Copy application files
COPY --chown=airflow:airflow . /app/

# Copy DAG files to Airflow directory
COPY --chown=airflow:airflow paste.txt /opt/airflow/dags/ml_pipeline_dag.py

# Create scripts directory structure
RUN mkdir -p /opt/airflow/dags/scripts && \
    ln -sf /app/extract.py /opt/airflow/dags/scripts/ && \
    ln -sf /app/cleaner.py /opt/airflow/dags/scripts/ && \
    ln -sf /app/loader.py /opt/airflow/dags/scripts/ && \
    ln -sf /app/dashboard.py /opt/airflow/dags/scripts/

# Set Airflow environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
ENV AIRFLOW__CORE__LOAD_EXAMPLES=false
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=true
ENV AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
ENV AIRFLOW__WEBSERVER__SECRET_KEY=your-secret-key-here

# AWS CLI for S3 operations
RUN pip install --no-cache-dir awscli boto3

# Create supervisor configuration
COPY <<EOF /etc/supervisor/conf.d/ml-pipeline.conf
[supervisord]
nodaemon=true
user=root

[program:airflow-webserver]
command=/usr/local/bin/airflow webserver --port 8080
user=airflow
environment=AIRFLOW_HOME=/opt/airflow
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/airflow/logs/webserver.log

[program:airflow-scheduler]
command=/usr/local/bin/airflow scheduler
user=airflow
environment=AIRFLOW_HOME=/opt/airflow
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/airflow/logs/scheduler.log

[program:airflow-triggerer]
command=/usr/local/bin/airflow triggerer
user=airflow
environment=AIRFLOW_HOME=/opt/airflow
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/airflow/logs/triggerer.log
EOF

# Create healthcheck script
COPY <<EOF /app/healthcheck.sh
#!/bin/bash
set -e

# Check Airflow webserver
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Airflow webserver healthy"
else
    echo "❌ Airflow webserver unhealthy"
    exit 1
fi

# Check if scheduler is running
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "✅ Airflow scheduler running"
else
    echo "❌ Airflow scheduler not running"
    exit 1
fi

echo "✅ All services healthy"
EOF

RUN chmod +x /app/healthcheck.sh

# Create initialization script
COPY <<EOF /app/init-airflow.sh
#!/bin/bash
set -e

echo "🚀 Initializing Airflow..."

# Initialize Airflow database
airflow db init

# Create admin user if it doesn't exist
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || echo "User already exists"

# Set up connections
echo "🔗 Setting up connections..."

# Neo4j connection
airflow connections add 'neo4j_default' \
    --conn-type 'generic' \
    --conn-host 'neo4j-server' \
    --conn-port 7687 \
    --conn-login 'neo4j' \
    --conn-password 'password' \
    --conn-extra '{"uri": "bolt://neo4j-server:7687"}' || echo "Neo4j connection exists"

# AWS connection
airflow connections add 'aws_default' \
    --conn-type 'aws' \
    --conn-extra '{"region_name": "us-east-1"}' || echo "AWS connection exists"

# Set up variables
echo "📝 Setting up variables..."
airflow variables set s3_bucket "ml-pipeline-bucket"
airflow variables set aws_region "us-east-1"
airflow variables set neo4j_uri "bolt://neo4j-server:7687"
airflow variables set dashboard_port "8501"

echo "✅ Airflow initialization complete"
EOF

RUN chmod +x /app/init-airflow.sh

# Create entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

# Default to airflow mode
MODE=\${MODE:-airflow}

echo "🚀 Starting ML Pipeline in \$MODE mode..."

case "\$MODE" in
    "airflow")
        echo "🔄 Starting Airflow services..."
        
        # Switch to airflow user
        su - airflow -c "cd /opt/airflow && /app/init-airflow.sh"
        
        # Start supervisor (which starts all Airflow services)
        exec /usr/bin/supervisord -c /etc/supervisor/conf.d/ml-pipeline.conf
        ;;
        
    "jenkins")
        echo "🔧 Starting in Jenkins mode..."
        
        # Switch to jenkins user and run the pipeline script
        exec su - jenkins -c "cd /app && bash -c '\$@'" -- "\$@"
        ;;
        
    "standalone")
        echo "🏃 Running standalone pipeline..."
        
        # Set environment variables
        export NEO4J_URI=\${NEO4J_URI:-bolt://neo4j-server:7687}
        export NEO4J_USERNAME=\${NEO4J_USERNAME:-neo4j}
        export NEO4J_PASSWORD=\${NEO4J_PASSWORD:-password}
        export AWS_DEFAULT_REGION=\${AWS_DEFAULT_REGION:-us-east-1}
        export S3_BUCKET=\${S3_BUCKET:-ml-pipeline-bucket}
        
        # Run the pipeline steps
        echo "📊 Step 1: Extracting data..."
        python /app/extract.py --output /tmp/raw_reviews.csv
        
        echo "🔄 Step 2: Processing data..."
        python /app/cleaner.py --input /tmp/raw_reviews.csv --output /tmp/processed_reviews.csv --topics 5
        
        echo "📤 Step 3: Loading to S3..."
        python /app/loader.py --file /tmp/processed_reviews.csv --bucket \$S3_BUCKET --key "data/processed_reviews_\$(date +%Y%m%d_%H%M%S).csv"
        
        echo "🎯 Step 4: Starting dashboard..."
        exec streamlit run /app/dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0
        ;;
        
    "dashboard")
        echo "📊 Starting dashboard only..."
        exec streamlit run /app/dashboard.py --server.headless true --server.port 8501 --server.address 0.0.0.0
        ;;
        
    *)
        echo "❌ Unknown mode: \$MODE"
        echo "Available modes: airflow, jenkins, standalone, dashboard"
        exit 1
        ;;
esac
EOF

RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8080 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Set working directory
WORKDIR /app

# Default to airflow mode
ENV MODE=airflow

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]