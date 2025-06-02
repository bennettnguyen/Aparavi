#!/bin/bash

# Enterprise RAG System Setup Script
echo "Setting up Enterprise RAG System..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✅ Docker check passed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker Compose check passed"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv aparavi
source aparavi/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check Qdrant health
echo "🔍 Checking Qdrant connection..."
if curl -s http://localhost:6333/health > /dev/null; then
    echo "✅ Qdrant is running"
else
    echo "⚠️ Qdrant may not be ready yet. Check with: curl http://localhost:6333/health"
fi

# Check Neo4j
echo "🔍 Checking Neo4j connection..."
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j is running"
    echo "🌐 Neo4j Browser: http://localhost:7474 (neo4j/password)"
else
    echo "⚠️ Neo4j may not be ready yet. Check at: http://localhost:7474"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key:"
echo "   export OPENAI_API_KEY='your_api_key_here'"
echo ""
echo "2. Run the application:"
echo "   streamlit run aparavi.py"
echo ""
echo "3. Open your browser to: http://localhost:8501"
echo ""
echo "For troubleshooting, check the README.md file." 