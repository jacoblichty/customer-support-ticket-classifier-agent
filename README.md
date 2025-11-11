# 🎫 Customer Support Ticket Classifier Agent

[![CI/CD Pipeline](https://github.com/jacoblichty/customer-support-ticket-classifier-agent/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/jacoblichty/customer-support-ticket-classifier-agent/actions)
[![codecov](https://codecov.io/gh/jacoblichty/customer-support-ticket-classifier-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/jacoblichty/customer-support-ticket-classifier-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991.svg?style=flat&logo=OpenAI)](https://openai.com)

An AI-powered system using OpenAI GPT-4 for automatically classifying customer support tickets into predefined categories, helping organizations improve response times and routing efficiency.

## 🚀 Features

- **🤖 AI-Powered Classification**: Uses OpenAI GPT-4 for intelligent ticket categorization
- **⚡ FastAPI REST API**: Production-ready API with automatic documentation
- **🔄 Batch Processing**: Concurrent processing of multiple tickets
- **🛡️ Fallback System**: Rule-based classification when AI is unavailable
- **📊 Real-time Statistics**: Processing metrics and category distribution
- **🏥 Health Monitoring**: Comprehensive health checks and monitoring endpoints
- **🐳 Docker Support**: Multi-stage Docker builds for development and production
- **🧪 Comprehensive Testing**: Unit tests with >90% coverage
- **📝 Structured Logging**: Configurable logging with rotation
- **⚙️ Environment Configuration**: Multiple environment support (dev/staging/prod)

## 📋 Supported Categories

- **Technical Issue**: Software bugs, hardware problems, system errors
- **Billing Inquiry**: Payment questions, invoice issues, subscription queries
- **Account Management**: Login problems, password resets, profile changes
- **Feature Request**: Enhancement suggestions, new feature requests
- **General Inquiry**: General questions, how-to requests
- **Complaint**: Service dissatisfaction, negative feedback
- **Refund Request**: Money-back requests, cancellations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Classifier    │    │   OpenAI API    │
│   Web Server    │───▶│   Engine        │───▶│   GPT-4         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Rule-based    │              │
         │              │   Fallback      │◀─────────────┘
         │              └─────────────────┘
         ▼
┌─────────────────┐
│   Monitoring    │
│   & Logging     │
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- Docker (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/jacoblichty/customer-support-ticket-classifier-agent.git
cd customer-support-ticket-classifier-agent
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the Application

#### Option A: Command Line Demo
```bash
# Run demo with sample tickets
python main.py --demo
```

#### Option B: Start API Server
```bash
# Start the FastAPI server
python main.py --server

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

#### Option C: Docker
```bash
# Build and run with Docker
docker build -t ticket-classifier .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here ticket-classifier

# Or use Docker Compose
echo "OPENAI_API_KEY=your_key_here" > .env
docker-compose up
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Classify Single Ticket
```http
POST /classify
Content-Type: application/json

{
  "ticket_id": "T001",
  "subject": "Login Issue",
  "content": "I can't log into my account",
  "customer_email": "user@example.com",
  "priority": "high"
}
```

#### Classify Multiple Tickets
```http
POST /classify/batch
Content-Type: application/json

{
  "tickets": [
    {
      "ticket_id": "T001",
      "subject": "Login Issue",
      "content": "I can't log into my account",
      "customer_email": "user@example.com",
      "priority": "high"
    }
  ]
}
```

#### Get Statistics
```http
GET /stats
```

#### Health Check
```http
GET /health
```

#### Available Categories
```http
GET /categories
```

### Example Response
```json
{
  "ticket_id": "T001",
  "subject": "Login Issue",
  "content": "I can't log into my account",
  "customer_email": "user@example.com",
  "priority": "high",
  "category": "account_management",
  "confidence_score": 0.95,
  "reasoning": "Keywords indicate account access issues",
  "created_at": "2025-11-11T10:30:00",
  "processed_at": "2025-11-11T10:30:01"
}
```

## 🛠️ Development

### Project Structure
```
├── src/
│   └── ticket_classifier/
│       ├── __init__.py
│       ├── models.py          # Data models and schemas
│       ├── classifier.py      # OpenAI classification logic
│       ├── agent.py          # Main processing agent
│       ├── api.py            # FastAPI application
│       ├── config.py         # Configuration management
│       └── logging_config.py # Logging setup
├── tests/                    # Unit and integration tests
├── main.py                   # Application entry point
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Development and production setup
├── requirements.txt         # Python dependencies
└── pyproject.toml          # Project configuration
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_classifier.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🐳 Docker Deployment

### Development
```bash
docker-compose --profile dev up --build
```

### Production
```bash
docker-compose up --build -d
```

### With Monitoring Stack
```bash
docker-compose --profile production --profile monitoring up -d
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model to use |
| `ENVIRONMENT` | `development` | Environment: development/production/testing |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_BATCH_SIZE` | `100` | Maximum tickets per batch |
| `MAX_CONCURRENT_REQUESTS` | `10` | Concurrent OpenAI requests |

### Configuration Files

- `.env` - Local development environment variables
- `.env.production` - Production environment template
- `pyproject.toml` - Project configuration and tool settings

## 📊 Monitoring & Observability

### Health Checks
- **Endpoint**: `GET /health`
- **Docker**: Built-in healthcheck
- **Kubernetes**: Readiness and liveness probes

### Metrics
- Processing statistics at `/stats`
- Recent tickets at `/recent`
- OpenAI API availability status

### Logging
- Structured JSON logging in production
- Configurable log levels
- Log rotation (10MB, 5 files)
- Console and file output

## 🚀 Production Deployment

### Docker Production Stack
```bash
# 1. Set up environment
cp .env.production .env
# Edit .env with your production values

# 2. Deploy with all services
docker-compose --profile production up -d

# 3. Monitor logs
docker-compose logs -f ticket-classifier
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml (example)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ticket-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ticket-classifier
  template:
    metadata:
      labels:
        app: ticket-classifier
    spec:
      containers:
      - name: ticket-classifier
        image: ghcr.io/jacoblichty/customer-support-ticket-classifier-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Docker Tests**: Container functionality
- **Load Tests**: Performance validation

### Running Different Test Suites
```bash
# Unit tests only
python -m pytest tests/test_*.py -m unit

# Integration tests
python -m pytest tests/test_api.py -m integration

# Docker integration test
docker build --target testing -t ticket-classifier:test .
docker run --rm ticket-classifier:test
```

## 🔒 Security

### API Security
- Input validation with Pydantic
- Rate limiting (configurable)
- CORS configuration
- Health check endpoints

### Container Security
- Non-root user execution
- Minimal base image (Python slim)
- Multi-stage builds
- Vulnerability scanning with Trivy

### Environment Security
- Environment-based configuration
- Secret management support
- Secure defaults

## 📈 Performance

### Benchmarks
- **Single Classification**: ~1-2 seconds with OpenAI
- **Batch Processing**: Concurrent requests (configurable)
- **Fallback Mode**: ~10ms rule-based classification
- **Memory Usage**: ~50-100MB base usage

### Optimization Tips
1. Use batch processing for multiple tickets
2. Configure `MAX_CONCURRENT_REQUESTS` based on your OpenAI limits
3. Enable rule-based fallback for reliability
4. Use caching for repeated classifications (Redis)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest`)
6. Run code quality checks (`black`, `flake8`, `mypy`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run all quality checks
make lint test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [OpenAI](https://openai.com/) for GPT-4 API
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [Docker](https://www.docker.com/) for containerization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/jacoblichty/customer-support-ticket-classifier-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jacoblichty/customer-support-ticket-classifier-agent/discussions)
- **Email**: jacob.lichty@example.com

---

Made with ❤️ by [Jacob Lichty](https://github.com/jacoblichty)