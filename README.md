# SignEase MVP 🤟 - AI-Powered Sign Language Bridge

**Real-time American Sign Language (ASL) Recognition System**

SignEase is a revolutionary web application that bridges communication gaps between deaf/hard-of-hearing individuals and the hearing community through real-time ASL gesture recognition with 99.57% accuracy, powered by custom-trained neural networks and GPU acceleration.

![SignEase Demo](https://img.shields.io/badge/Demo-Live-brightgreen) ![Version](https://img.shields.io/badge/Version-1.0.0-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Accuracy](https://img.shields.io/badge/Accuracy-99.57%25-success) ![Tests](https://img.shields.io/badge/Tests-79%20Passing-brightgreen)

## 🌟 Features

### Core Functionality
- **Real-time ASL Recognition**: 99.57% accuracy with custom-trained ML model
- **Immersive AR Overlay**: Text appears directly over hand gestures
- **Text-to-Speech**: Natural voice synthesis with customizable settings
- **Sentence Building**: Intelligent word completion and sentence construction
- **Multi-language Support**: English, Spanish, French interface options

### Advanced Features
- **Performance Optimization**: GPU acceleration, request batching, ML inference optimization
- **Professional UI/UX**: Dark/light themes, responsive design, accessibility features
- **Comprehensive Testing**: 79 automated tests covering E2E, performance, and accuracy
- **Demo Environment**: Professional presentation tools with environment validation
- **Enterprise Monitoring**: Real-time performance metrics and optimization suggestions

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+ with pip
- Modern web browser with camera access
- GPU recommended for optimal performance

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/signease-mvp.git
   cd signease-mvp
   ```

2. **Setup Frontend**
   ```bash
   cd signease-frontend
   npm install
   npm run dev
   ```

3. **Setup Backend**
   ```bash
   cd ../backend
   pip install -r requirements.txt
   python app.py
   ```

4. **Access the Application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Environment Variables

Create `.env` files in both frontend and backend directories:

**Frontend (.env)**
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=SignEase MVP
VITE_APP_VERSION=1.0.0
VITE_ENABLE_PERFORMANCE_MONITORING=true
```

**Backend (.env)**
```env
MODEL_PATH=./models/asl_model_best_20251102_214717.json
ENABLE_GPU=true
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173,https://signease-mvp.vercel.app
```

## 🎯 Live Demo

**🌐 [Try SignEase Live](https://signease-mvp.vercel.app)**

### Demo Instructions
1. Allow camera permissions when prompted
2. Position your hand in the camera view
3. Sign ASL letters (A-Z supported)
4. Watch real-time recognition with AR overlay
5. Use sentence builder for complete words
6. Enable text-to-speech for voice output

### Demo Features to Try
- Spell "HELLO" in ASL
- Toggle AR overlay modes (floating, fixed, following)
- Adjust confidence thresholds
- Try different lighting conditions
- Test speech synthesis with various voices
- Explore performance monitoring dashboard

## 📊 Datasets & Models

SignEase uses two different datasets and corresponding trained models to provide flexible ASL recognition capabilities:

### Dataset 1: Lightweight Model (76K Parameters)
- **Parameters**: 76,000 trainable parameters
- **Source**: Custom curated dataset (included in project)
- **Size**: Optimized for quick inference and lower resource usage
- **Use Case**: Real-time recognition on devices with limited GPU/CPU resources
- **Accuracy**: ~95-97% on validation set
- **Inference Speed**: <50ms per prediction
- **Model File**: `best_asl_model.pth` (included in repository)
- **Training Scripts**: 
  - `simple_tensorflow_trainer.py`
  - `sklearn_asl_trainer.py`
  - `improved_sklearn_trainer.py`

### Dataset 2: High-Accuracy Model (3M Parameters)
- **Parameters**: 3,000,000 trainable parameters
- **Source**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Dataset Details**:
  - 87,000+ images of ASL alphabet signs
  - 29 classes (A-Z letters + SPACE, DELETE, NOTHING)
  - 200x200 RGB images
  - Multiple hand positions and lighting conditions
  - Diverse backgrounds and skin tones
- **Use Case**: Maximum accuracy for production environments with adequate GPU resources
- **Accuracy**: 99.57% on validation set
- **Inference Speed**: ~100ms per prediction
- **Model File**: `asl_model_best_20251102_214717.json` (in backend/models/)
- **Training Scripts**:
  - `gpu_train_asl.py`
  - `tensorflow_gpu_asl_trainer.py`
  - `pytorch_ultimate_gpu_trainer.py`
  - `rtx5060_full_trainer.py`
  - `max_gpu_utilization_trainer.py`

### Dataset Comparison

| Feature | Lightweight (76K) | High-Accuracy (3M) |
|---------|-------------------|-------------------|
| Parameters | 76,000 | 3,000,000 |
| Model Size | ~300KB | ~12MB |
| Training Time | ~10 minutes (CPU) | ~2-4 hours (GPU) |
| Inference Speed | <50ms | ~100ms |
| Accuracy | 95-97% | 99.57% |
| GPU Required | No | Recommended |
| Memory Usage | <256MB | <512MB |
| Best For | Edge devices, demos | Production, high accuracy |

### Downloading the Kaggle Dataset

To train the high-accuracy model yourself:

1. **Install Kaggle CLI**
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle API Credentials**
   - Go to https://www.kaggle.com/account
   - Create new API token (downloads `kaggle.json`)
   - Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

3. **Download Dataset**
   ```bash
   kaggle datasets download -d grassknoted/asl-alphabet
   unzip asl-alphabet.zip -d ./data/asl-alphabet
   ```

4. **Train Model**
   ```bash
   # For GPU training (recommended)
   python gpu_train_asl.py
   
   # For maximum GPU utilization
   python max_gpu_utilization_trainer.py
   
   # For RTX 5060 optimized training
   python rtx5060_full_trainer.py
   ```

### Model Selection Guide

**Use Lightweight Model (76K) when:**
- Deploying to edge devices or mobile
- Limited GPU/CPU resources available
- Quick inference time is critical
- Running demos or prototypes
- Bandwidth/storage is constrained

**Use High-Accuracy Model (3M) when:**
- Maximum accuracy is required
- GPU resources are available
- Production environment with quality requirements
- Handling diverse lighting/background conditions
- Need for robust real-world performance

### Training Your Own Models

Both models can be retrained with custom data:

```bash
# Lightweight model training
python simple_tensorflow_trainer.py --epochs 50 --batch-size 32

# High-accuracy model training (requires GPU)
python gpu_train_asl.py --epochs 100 --batch-size 64 --data-path ./data/asl-alphabet

# Monitor training with GPU utilization
python monitor_system.py
```

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   ML Pipeline   │
│   (React/TS)    │◄──►│   (FastAPI)      │◄──►│   (TensorFlow)  │
│                 │    │                  │    │                 │
│ • Camera Input  │    │ • Gesture API    │    │ • Hand Tracking │
│ • AR Overlay    │    │ • Health Check   │    │ • ASL Model     │
│ • Speech Output │    │ • Performance    │    │ • Optimization  │
│ • UI/UX         │    │ • Documentation  │    │ • Inference     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Technology Stack

**Frontend**
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite with HMR
- **Styling**: Tailwind CSS with custom design system
- **State Management**: React Context + Custom hooks
- **Camera**: MediaPipe Hands for hand tracking
- **Performance**: Custom optimization engine with GPU profiling
- **Testing**: Vitest with 79 comprehensive tests

**Backend**
- **Framework**: FastAPI with async/await
- **ML Framework**: TensorFlow 2.x with custom model
- **Performance**: GPU acceleration with CUDA support
- **API**: RESTful with automatic OpenAPI documentation
- **Monitoring**: Custom performance tracking and optimization

**ML Pipeline**
- **Hand Tracking**: MediaPipe Hands (21 landmark points)
- **Model**: Custom CNN trained on ASL dataset
- **Accuracy**: 99.57% on validation set
- **Optimization**: Model quantization and inference batching
- **Real-time**: <100ms end-to-end latency

## 📊 Performance Metrics

### Accuracy Benchmarks
- **Overall Accuracy**: 99.57%
- **Per-letter Average**: 98.2%
- **Confidence Threshold**: 70%
- **False Positive Rate**: <2%

### Performance Benchmarks
- **End-to-end Latency**: <100ms
- **Frame Rate**: 30 FPS sustained
- **Memory Usage**: <512MB typical
- **GPU Memory**: <2GB with optimization
- **API Response Time**: <50ms average

### Browser Compatibility
- ✅ Chrome 90+ (Recommended)
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ⚠️ Mobile browsers (limited performance)

## 🧪 Testing

### Test Coverage
- **79 Total Tests** across 5 categories
- **E2E Tests**: Complete user workflows
- **Integration Tests**: Component interactions
- **Performance Tests**: Speed and memory validation
- **Accuracy Tests**: ML model validation
- **Cross-browser Tests**: Compatibility validation

### Running Tests
```bash
# Run all tests
npm run test

# Run specific test suites
npm run test:e2e          # End-to-end tests
npm run test:integration  # Integration tests
npm run test:performance  # Performance tests
npm run test:accuracy     # Accuracy validation
npm run test:browser      # Cross-browser tests

# Generate coverage report
npm run test:coverage
```

## 🚀 Deployment

### Vercel Deployment (Recommended)

1. **Connect Repository**
   ```bash
   # Install Vercel CLI
   npm i -g vercel
   
   # Deploy
   vercel --prod
   ```

2. **Environment Configuration**
   - Set environment variables in Vercel dashboard
   - Configure custom domain if needed
   - Enable analytics and monitoring

3. **Backend Deployment**
   - Deploy backend to Vercel Functions or separate service
   - Update CORS origins for production domain
   - Configure SSL certificates

### Manual Deployment

1. **Build for Production**
   ```bash
   npm run build
   ```

2. **Deploy Static Files**
   - Upload `dist/` folder to your hosting provider
   - Configure server for SPA routing
   - Set up HTTPS and security headers

## 🔧 Development

### Project Structure
```
signease-mvp/
├── signease-frontend/          # React frontend application
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Utility functions
│   │   ├── styles/            # CSS and theme files
│   │   └── tests/             # Test files
│   ├── public/                # Static assets
│   └── dist/                  # Build output
├── backend/                   # FastAPI backend
│   ├── api/                   # API routes
│   ├── models/                # ML models
│   ├── utils/                 # Utility functions
│   └── tests/                 # Backend tests
└── docs/                      # Documentation
```

### Development Commands

**Frontend Development**
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

**Backend Development**
```bash
python app.py        # Start development server
python -m pytest    # Run backend tests
python train.py      # Retrain ML model
python optimize.py   # Optimize model performance
```

### Code Quality

- **TypeScript**: Strict type checking enabled
- **ESLint**: Airbnb configuration with custom rules
- **Prettier**: Automatic code formatting
- **Husky**: Pre-commit hooks for quality checks
- **Conventional Commits**: Standardized commit messages

## 📚 API Documentation

### Core Endpoints

#### Gesture Recognition
```http
POST /api/predict
Content-Type: application/json

{
  "landmarks": [[x1, y1, z1], [x2, y2, z2], ...],
  "confidence_threshold": 0.7
}
```

**Response**
```json
{
  "prediction": "A",
  "confidence": 0.95,
  "alternatives": [
    {"prediction": "S", "confidence": 0.12},
    {"prediction": "T", "confidence": 0.08}
  ],
  "processing_time": 45.2
}
```

#### Health Check
```http
GET /api/health
```

**Response**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "gpu_available": true,
  "uptime": 3600.5
}
```

#### Performance Metrics
```http
GET /api/metrics
```

**Response**
```json
{
  "requests_per_second": 25.3,
  "average_latency": 47.8,
  "model_accuracy": 0.9957,
  "gpu_memory_usage": 1.2,
  "cache_hit_rate": 0.85
}
```

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎨 UI/UX Features

### Design System
- **Modern Interface**: Clean, professional design
- **Dark/Light Themes**: Automatic and manual switching
- **Responsive Design**: Works on desktop, tablet, mobile
- **Accessibility**: WCAG 2.1 AA compliant
- **Animations**: Smooth transitions and micro-interactions

### User Experience
- **Intuitive Controls**: Easy-to-use interface
- **Real-time Feedback**: Immediate visual confirmation
- **Error Handling**: Graceful error messages and recovery
- **Performance Monitoring**: Built-in performance dashboard
- **Customization**: Extensive settings and preferences

## 🔒 Security & Privacy

### Security Features
- **HTTPS Only**: Secure communication
- **CSP Headers**: Content Security Policy protection
- **Input Validation**: Server-side validation for all inputs
- **Rate Limiting**: API abuse prevention
- **Error Handling**: No sensitive information in error messages

### Privacy Protection
- **Local Processing**: Camera data never leaves the device
- **No Data Storage**: No personal data stored on servers
- **Minimal Logging**: Only essential metrics logged
- **Transparent**: Open source for full transparency

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Standards
- Follow TypeScript/Python best practices
- Write comprehensive tests
- Document new features
- Follow conventional commit format
- Ensure accessibility compliance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team**: For excellent hand tracking technology
- **TensorFlow Team**: For powerful ML framework
- **ASL Community**: For inspiration and feedback
- **Open Source Contributors**: For various libraries and tools

## 📞 Support

### Getting Help
- **Documentation**: Check this README and inline documentation
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: contact@signease.dev

### Reporting Issues
When reporting issues, please include:
- Browser and version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Console errors (if any)

---

**Made with ❤️ for the deaf and hard-of-hearing community**

*SignEase MVP - Breaking down communication barriers through technology*
