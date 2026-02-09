# Running Benchmarks on AWS

This guide explains how to run the Wikipedia benchmark on AWS with a machine that has 8GB RAM.

## Quick Start

### 1. Launch an EC2 Instance

Region: `eu-west-1` (Ireland)
Key: `razer.pem` (already in your ~/.ssh/)

```bash
# Using AWS CLI (eu-west-1)
aws ec2 run-instances \
  --region eu-west-1 \
  --image-id ami-07b2ea531b7c69e92 \
  --instance-type t3.large \
  --key-name razer \
  --security-group-ids sg-xxxxxxxx \
  --subnet-id subnet-xxxxxxxx

# SSH into the instance
ssh -i ~/.ssh/razer.pem ubuntu@<public-ip>
```

Or use the AWS Console:
- Region: eu-west-1 (Ireland)
- AMI: Ubuntu 24.04 LTS (ami-07b2ea531b7c69e62 in eu-west-1)
- Instance Type: `t3.large` (2 vCPU, 8 GB RAM)
- Storage: gp3 (20 GB)

### 2. Install uv and Run Benchmark

```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repo
git clone https://github.com/alexeygrigorev/minsearch.git
cd minsearch/benchmark

# Download and parse Wikipedia
uv run python download_wikipedia.py
uv run python parse_wikipedia.py data/simplewiki-*.xml.bz2

# Run benchmark
uv run python run_full_benchmark.py
```

## Instance Type Recommendations

| Instance Type | vCPU | Memory | Cost/hr | Use Case |
|---------------|----- |--------|---------|----------|
| t3.large | 2 | 8 GB | ~$0.08 | Up to ~125K documents |
| t3.xlarge | 4 | 16 GB | ~$0.16 | Recommended for full dataset |
| m6i.xlarge | 4 | 16 GB | ~$0.13 | Better CPU for full dataset |

## Cost Estimate

- t3.large: ~$0.08/hour (up to 125K docs)
- t3.xlarge: ~$0.16/hour (full 291K docs)
- Benchmark runtime: ~20-60 minutes
- Total cost: ~$0.03-$0.16 per run

## What the Benchmark Does

1. Downloads Simple Wikipedia dump (~444MB compressed)
2. Parses XML to JSONL (291,737 documents, ~1GB)
3. Indexes with both Regular Index and AppendableIndex
4. Benchmarks search performance with 10 random queries
5. Reports timing and QPS for both implementations

## Actual AWS Benchmark Results

Run on EC2 t3.large (eu-west-1, 2 vCPU, 8GB RAM):

### 125K Documents (fits in 8GB)
- Indexing: Regular=57.76s, Appendable=79.08s (1.37x slower)
- Search: Regular=1043.73ms (1.0 QPS), Appendable=13.66ms (73 QPS)
- Search is 76x FASTER

### Full Dataset (291K documents)
- Does NOT fit in 8GB RAM with both indices
- Use t3.xlarge (16GB) or m6i.xlarge for full dataset

### Key Findings
- Indexing overhead: ~1.3-1.4x slower (acceptable trade-off)
- Search performance: 50-76x FASTER than Regular Index
- Performance improves with dataset size due to better algorithmic efficiency
