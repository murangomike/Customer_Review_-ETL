# ML Pipeline Docker Setup Guide

This Docker setup provides a complete containerized environment for your ML pipeline that supports both Airflow orchestration and Jenkins CI/CD workflows.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Neo4j DB      │    │  ML Pipeline    │    │   PostgreSQL    │
│   (Data Source) │────│   (Airflow)     │────│  (Metadata)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │  Streamlit      │
                       │  (Dashboard)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Build the production image
docker build -t ml-pipeline:latest .

# Or build specific stage
docker build --target airflow-base -t ml-pipeline:airflow .
```

### 2. Run with Docker Compose

```bash
# Start all services (Airflow mode)
docker-compose up -d

# Start in standalone mode
docker-compose --profile standalone up -d ml-pipeline-standalone

# Start only dashboard
docker-compose --profile dashboard up -d ml-dashboard

# Start with local development tools
docker-compose --profile dev --profile local up -d
```

### 3. Access Services

- **Airflow Web UI**: http://localhost:8080 (admin/admin)
- **Streamlit Dashboard**: http://localhost:8501
- **Neo4j Browser**: http://localhost:7474 (neo4j/password)
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Jupyter Notebook**: http://localhost:8888

## 🔧 Configuration Modes

### Airflow Mode (Default)
```bash
docker run -e MODE=airflow -p 8080:8080 -p 8501:8501 ml-pipeline:latest
```
- Runs full Airflow scheduler and webserver
- Executes DAGs on schedule
- Provides web UI for monitoring

### Jenkins Mode
```bash
docker run -e MODE=jenkins ml-pipeline:latest python extract.py
```
- Optimized for CI/CD pipeline execution
- No persistent services, just runs the pipeline

### Standalone Mode
```bash
docker run -e MODE=standalone \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e S3_BUCKET=your-bucket \
  ml-pipeline:latest
```
- Runs complete pipeline once
- Starts dashboard at the end
- Good for batch processing

### Dashboard Only Mode
```bash
docker run -e MODE=dashboard -p 8501:8501 ml-pipeline:latest
```
- Only runs the Streamlit dashboard
- Expects data files to be available

## 🔐 Environment Variables

### Required
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `S3_BUCKET`: AWS S3 bucket name

### Optional
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `DASHBOARD_PORT`: Streamlit port (default: 8501)
- `AIRFLOW__CORE__FERNET_KEY`: Airflow encryption key

## 📁 Volume Mounts

### For Development
```bash
docker-compose up -d
```
Mounts:
- `./dags:/opt/airflow/dags` - Airflow DAGs
- `./scripts:/opt/airflow/dags/scripts` - Pipeline scripts
- `./logs:/opt/airflow/logs` - Airflow logs
- `./data:/app/data` - Data files

### For Production
```bash
docker run -v /host/data:/app/data ml-pipeline:latest
```

## 🔍 Monitoring and Debugging

### Check Service Health
```bash
# All services
docker-compose ps

# Specific service logs
docker-compose logs ml-pipeline
docker-compose logs neo4j

# Follow logs
docker-compose logs -f ml-pipeline
```

### Health Checks
```bash
# Manual health check
docker exec ml-pipeline-airflow /app/healthcheck.sh

# Check Airflow components
docker exec ml-pipeline-airflow airflow jobs check --health-check
```

### Debug Mode
```bash
# Run with debug shell
docker run -it --entrypoint /bin/bash ml-pipeline:latest

# Check Airflow status
docker exec -it ml-pipeline-airflow airflow jobs check --health-check
```

## 🔄 Common Operations

### Initialize Airflow
```bash
# First time setup
docker exec ml-pipeline-airflow /app/init-airflow.sh

# Create additional users
docker exec ml-pipeline-airflow airflow users create \
  --username user --password pass --firstname First --lastname Last \
  --role Viewer --email user@example.com
```

### Trigger DAG Manually
```bash
# Trigger ML pipeline DAG
docker exec ml-pipeline-airflow airflow dags trigger ml_pipeline_etl_analysis

# Check DAG status
docker exec ml-pipeline-airflow airflow dags state ml_pipeline_etl_analysis
```

### Data Operations
```bash
# Extract data manually
docker exec ml-pipeline-airflow python /app/extract.py --output /tmp/data.csv

# Process data
docker exec ml-pipeline-airflow python /app/cleaner.py --input /tmp/data.csv --output /tmp/processed.csv

# Upload to S3
docker exec ml-pipeline-airflow python /app/loader.py --file /tmp/processed.csv --bucket your-bucket
```

## 🐛 Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```bash
   # Check Neo4j is running
   docker-compose ps neo4j
   
   # Check connection
   docker exec ml-pipeline-neo4j cypher-shell -u neo4j -p password "RETURN 1"
   ```

2. **Airflow Web UI Not Loading**
   ```bash
   # Check webserver logs
   docker-compose logs ml-pipeline
   
   # Restart webserver
   docker-compose restart ml-pipeline
   ```

3. **S3 Upload Failures**
   ```bash
   # Check AWS credentials
   docker exec ml-pipeline-airflow aws sts get-caller-identity
   
   # Test S3 access
   docker exec ml-pipeline-airflow aws s3 ls s3://your-bucket
   ```

4. **Dashboard Not Starting**
   ```bash
   # Check if data files exist
   docker exec ml-pipeline-airflow ls -la /app/
   
   # Manually start dashboard
   docker exec ml-pipeline-airflow streamlit run /app/dashboard.py
   ```

### Performance Tuning

1. **Memory Settings**
   ```yaml
   # In docker-compose.yml
   services:
     ml-pipeline:
       deploy:
         resources:
           limits:
             memory: 4G
           reservations:
             memory: 2G
   ```

2. **Neo4j Memory**
   ```yaml
   neo4j:
     environment:
       - NEO4J_dbms_memory_heap_max__size=4G
       - NEO4J_dbms_memory_pagecache_size=2G
   ```

## 🔧 Production Deployment

### Using Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ml-pipeline

# Scale services
docker service scale ml-pipeline_ml-pipeline=3
```

### Using Kubernetes
```bash
# Convert compose to k8s manifests
kompose convert -f docker-compose.yml

# Apply to cluster
kubectl apply -f .
```

### Health Monitoring
```bash
# Add monitoring to docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

## 📊 Metrics and Monitoring

The container includes built-in health checks and metrics:

- **Airflow**: Web UI provides DAG run metrics
- **Application**: Logs to stdout/stderr for container log collection
- **Health Checks**: Built-in endpoints for load balancer health checks
- **Metrics**: Pipeline execution metrics saved to `/tmp/model_metrics.json`

## 🚀 Next Steps

1. **Set up CI/CD**: Use the Jenkins