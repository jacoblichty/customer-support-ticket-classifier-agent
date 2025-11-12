# 🎫 Customer Support Ticket Classifier Agent

An AI-powered FastAPI service that uses Azure OpenAI GPT-4 to automatically classify customer support tickets into predefined categories, helping organizations improve response times and routing efficiency.

## 🚀 Features

- **🤖 AI-Powered Classification**: Uses Azure OpenAI GPT-4 for intelligent ticket categorization
- **⚡ FastAPI REST API**: Production-ready API with automatic documentation
- **🔄 Batch Processing**: Concurrent processing of multiple tickets
- **🛡️ Fallback System**: Rule-based classification when AI is unavailable
- ** Docker Support**: Multi-stage Docker builds optimized for Azure deployment
- **🧪 Comprehensive Testing**: 74 unit tests with 100% pass rate
- **📝 Structured Logging**: Production-ready logging with rotation

## 📋 Supported Categories

- **Technical Issue**: Software bugs, hardware problems, system errors
- **Billing Inquiry**: Payment questions, invoice issues, subscription queries
- **Account Management**: Login problems, password resets, profile changes
- **Feature Request**: Enhancement suggestions, new feature requests
- **General Inquiry**: General questions, how-to requests
- **Complaint**: Service dissatisfaction, negative feedback
- **Refund Request**: Money-back requests, cancellations

## 🏗️ Project Structure

```
├── src/ticket_classifier/
│   ├── models.py          # Data models and schemas
│   ├── classifier.py      # OpenAI classification logic
│   ├── agent.py          # Main processing agent
│   ├── api.py            # FastAPI application
│   └── config.py         # Configuration management
├── tests/                # Unit tests (74 tests)
├── main.py              # Application entry point
├── Dockerfile           # Multi-stage Docker build
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start with Docker

### Prerequisites

- Docker
- Azure OpenAI resource with a deployed GPT-4 model

### Environment Setup

Create a `.env` file with your Azure OpenAI configuration:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt4-deployment-name
```

### Docker Commands

#### Development Stage

Run with hot reload and development features:

```bash
docker build --target development -t ticket-classifier:dev .
docker run -p 8000:8000 --env-file .env -v ${PWD}/src:/app/src ticket-classifier:dev
```

#### Testing Stage

Run unit tests inside the container:

```bash
docker build --target testing -t ticket-classifier:test .
docker run --rm --env-file .env ticket-classifier:test
```

#### Production Stage

Build and run the optimized production container:

```bash
docker build --target production -t ticket-classifier:prod .
docker run -p 8000:8000 --env-file .env ticket-classifier:prod
```

### Access the API

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📚 API Endpoints

### Main Endpoints

- `POST /classify` - Classify a single ticket
- `POST /classify/batch` - Classify multiple tickets
- `GET /health` - Health check
- `GET /stats` - Processing statistics
- `GET /categories` - Available categories
- `GET /docs` - Interactive API documentation

### Example Request

```json
{
  "ticket_id": "T001",
  "subject": "Login Issue",
  "content": "I can't log into my account",
  "customer_email": "user@example.com",
  "priority": "high"
}
```

### Example Response

```json
{
  "ticket_id": "T001",
  "category": "account_management",
  "confidence_score": 0.95,
  "reasoning": "Keywords indicate account access issues",
  "processed_at": "2025-11-11T10:30:01"
}
```

## 🧪 Testing

### Run Tests with Docker

```bash
# Run all 74 unit tests (tests run automatically during build)
docker build --target testing -t ticket-classifier:test .

# Run tests with coverage report
docker run --rm ticket-classifier:test pytest --cov=src --cov-report=html
```

### Local Development (Optional)

If you prefer local development:

```bash
# Set up virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies and run tests
pip install -r requirements.txt
pytest
```

## ☁️ Azure Deployment

This application is optimized for Azure Container Apps or Azure App Service.

### Azure Container Apps Example

```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image ticket-classifier:latest .

# Deploy to Azure Container Apps
az containerapp create \
  --name ticket-classifier \
  --resource-group myResourceGroup \
  --environment myContainerEnvironment \
  --image myregistry.azurecr.io/ticket-classifier:latest \
  --env-vars OPENAI_API_KEY=secretref:openai-key \
  --ingress external --target-port 8000
```

## ⚙️ Configuration

### Required Environment Variables

- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI API key (required)
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint URL (required)
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Your deployed model name (required)

### Optional Environment Variables

- `AZURE_OPENAI_API_VERSION` - API version (default: `2024-02-15-preview`)
- `ENVIRONMENT` - Environment: development/production/testing
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `MAX_BATCH_SIZE` - Maximum tickets per batch (default: `100`)
- `MAX_CONCURRENT_REQUESTS` - Concurrent requests (default: `10`)

## License

This project is licensed under the MIT License.
