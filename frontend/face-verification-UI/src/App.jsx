import React, { useState, useCallback, useEffect } from 'react';
import { Upload, Play, BarChart3, CheckCircle, XCircle, ImageIcon } from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000'; // Adjust this to your FastAPI server URL

const ImageDropzone = ({ onImageSelect, image, label }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      onImageSelect(files[0]);
    }
  }, [onImageSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileSelect = useCallback((e) => {
    const files = e.target.files;
    if (files.length > 0) {
      onImageSelect(files[0]);
    }
  }, [onImageSelect]);

  return (
    <div className="w-full">
      <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
      <div
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${dragOver
          ? 'border-blue-500 bg-blue-50'
          : image
            ? 'border-green-500 bg-green-50'
            : 'border-gray-300 hover:border-gray-400'
          }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        {image ? (
          <div className="space-y-2">
            <img
              src={URL.createObjectURL(image)}
              alt="Preview"
              className="mx-auto h-32 w-32 object-cover rounded-lg"
            />
            <p className="text-sm text-green-600 font-medium">{image.name}</p>
          </div>
        ) : (
          <div className="space-y-2">
            <ImageIcon className="mx-auto h-12 w-12 text-gray-400" />
            <div>
              <p className="text-gray-600">Drop an image here or click to select</p>
              <p className="text-xs text-gray-400 mt-1">PNG, JPG, JPEG up to 10MB</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ResultCard = ({ result }) => {
  const isMatch = result.label === 'same';
  const confidence = (result.pred_value * 100).toFixed(2);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Prediction Result</h3>
        <div className={`flex items-center space-x-2 ${isMatch ? 'text-green-600' : 'text-red-600'}`}>
          {isMatch ? <CheckCircle className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
          <span className="font-medium">{result.label}</span>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">Model:</span>
          <span className="font-medium text-gray-900">{result.model_name}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">Confidence:</span>
          <span className="font-medium text-gray-900">{confidence}%</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">Raw Score:</span>
          <span className="font-mono text-sm text-gray-900">{result.pred_value.toFixed(6)}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">Binary Output:</span>
          <span className={`font-bold ${isMatch ? 'text-green-600' : 'text-red-600'}`}>
            {result.binary}
          </span>
        </div>

        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Confidence Level</span>
            <span>{confidence}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${isMatch ? 'bg-green-500' : 'bg-red-500'
                }`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricsCard = ({ metric }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{metric.model}</h3>
        <BarChart3 className="h-5 w-5 text-blue-500" />
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">Accuracy:</span>
          <span className="font-medium text-green-600">{(metric.accuracy * 100).toFixed(2)}%</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">AUC:</span>
          <span className="font-medium text-blue-600">{metric.auc.toFixed(4)}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">MAE:</span>
          <span className="font-medium text-orange-600">{metric.mae.toFixed(4)}</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-gray-600">MSE:</span>
          <span className="font-medium text-red-600">{metric.mse.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
};

export default function FaceRecognitionApp() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [selectedModel, setSelectedModel] = useState('concat_siamese_model.keras');
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [error, setError] = useState(null);

  const models = [
    'concat_siamese_model.keras',
    'cosine_metrics_model.keras',
    'classification_model.keras'
  ];

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    setMetricsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error('Error fetching metrics:', err);
      setError('Failed to load model metrics');
    } finally {
      setMetricsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!image1 || !image2) {
      setError('Please select both images');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = new FormData();
      formData.append('image1', image1);
      formData.append('image2', image2);
      formData.append('model', selectedModel);

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Failed to make prediction. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Face Recognition System</h1>
          <p className="text-gray-600">Upload two images to compare using AI models</p>
        </div>

        {/* Prediction Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Image Comparison</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <ImageDropzone
              onImageSelect={setImage1}
              image={image1}
              label="First Image"
            />
            <ImageDropzone
              onImageSelect={setImage2}
              image={image2}
              label="Second Image"
            />
          </div>

          <div className="flex flex-col sm:flex-row gap-4 items-end mb-6">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {models.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>

            <button
              onClick={handlePredict}
              disabled={loading || !image1 || !image2}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-colors"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Test Images</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-600">{error}</p>
            </div>
          )}

          {prediction && (
            <div className="mb-6">
              <ResultCard result={prediction} />
            </div>
          )}
        </div>

        {/* Metrics Section */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Model Performance Metrics</h2>
            <button
              onClick={fetchMetrics}
              disabled={metricsLoading}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              {metricsLoading ? 'Loading...' : 'Refresh'}
            </button>
          </div>

          {metricsLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          ) : metrics.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {metrics.map((metric, index) => (
                <MetricsCard key={index} metric={metric} />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No metrics data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
}