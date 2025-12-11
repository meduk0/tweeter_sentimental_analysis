#!/bin/bash

# Twitter Sentiment Analysis Web App - Quick Start Script

echo "======================================"
echo "Twitter Sentiment Analysis Web App"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if models directory exists
if [ ! -d "../models" ]; then
    echo -e "${RED}❌ Error: '../models' directory not found!${NC}"
    echo ""
    echo "Please run the Jupyter notebook first to train and save the models:"
    echo "  1. Open: jupyter notebook ../notebooks/TwitterSentimentAnalysis_Spark.ipynb"
    echo "  2. Run all cells, especially Section 12 (Save Best Model)"
    echo "  3. Then run this script again"
    echo ""
    exit 1
fi

# Check if dashboard_data.json exists
if [ ! -f "../models/dashboard_data.json" ]; then
    echo -e "${YELLOW}⚠️  Warning: 'dashboard_data.json' not found!${NC}"
    echo "The dashboard tab may not work properly."
    echo "Please ensure you ran all cells in Section 12 of the notebook."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✓ Model files found${NC}"
echo ""

# Ask user how to run
echo "How would you like to run the application?"
echo "  1) Docker Compose (recommended)"
echo "  2) Local Python environment"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Starting with Docker Compose..."
        echo ""
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}❌ Docker is not installed!${NC}"
            echo "Please install Docker first: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        # Check if Docker Compose is installed
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}❌ Docker Compose is not installed!${NC}"
            echo "Please install Docker Compose first."
            exit 1
        fi
        
        echo -e "${GREEN}✓ Docker and Docker Compose found${NC}"
        echo ""
        echo "Building and starting the container..."
        echo "(This may take a few minutes on first run)"
        echo ""
        
        docker-compose up --build
        ;;
        
    2)
        echo ""
        echo "Starting with local Python environment..."
        echo ""
        
        # Check if Python is installed
        if ! command -v python3 &> /dev/null; then
            echo -e "${RED}❌ Python 3 is not installed!${NC}"
            exit 1
        fi
        
        # Check if Java is installed
        if ! command -v java &> /dev/null; then
            echo -e "${YELLOW}⚠️  Warning: Java not found!${NC}"
            echo "PySpark requires Java 11 or 17."
            echo "Please install Java first."
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        # Check if requirements are installed
        if ! python3 -c "import streamlit" 2>/dev/null; then
            echo -e "${YELLOW}⚠️  Required packages not found. Installing...${NC}"
            pip install -r requirements.txt
        fi
        
        echo -e "${GREEN}✓ Python environment ready${NC}"
        echo ""
        echo "Starting Streamlit app..."
        echo ""
        
        streamlit run app.py
        ;;
        
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac
