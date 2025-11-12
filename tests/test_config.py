"""
Tests for configuration management.
"""

import pytest
import os
from unittest.mock import patch

from ticket_classifier.config import (
    Settings, UnitTestSettings, DevelopmentSettings, 
    ProductionSettings, get_settings, get_environment_settings
)


class TestSettings:
    """Test cases for Settings class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        settings = Settings()
        
        assert settings.app_name == "Customer Support Ticket Classifier API"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.openai_model == "gpt-4"
        assert settings.openai_temperature == 0.3
        assert settings.max_batch_size == 100
        assert settings.log_level == "INFO"
    

    

    

    
    def test_cors_origins_parsing(self):
        """Test parsing CORS origins from various formats."""
        # Test string parsing
        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://localhost:3000,https://example.com'}):
            settings = Settings()
            expected = ["http://localhost:3000", "https://example.com"]
            assert settings.cors_origins == expected
        
        # Test list format
        origins = ["http://localhost:3000", "https://example.com"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins
    
    def test_ticket_categories_parsing(self):
        """Test parsing ticket categories from various formats."""
        # Test string parsing
        categories_str = "technical,billing,general"
        settings = Settings(ticket_categories=categories_str)
        expected = ["technical", "billing", "general"]
        assert settings.ticket_categories == expected
        
        # Test list format
        categories = ["technical", "billing", "general"]
        settings = Settings(ticket_categories=categories)
        assert settings.ticket_categories == categories


class TestEnvironmentSpecificSettings:
    """Test cases for environment-specific settings."""
    
    def test_development_settings(self):
        """Test development environment settings."""
        settings = DevelopmentSettings()
        
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.host == "127.0.0.1"
        assert settings.workers == 1
    
    def test_production_settings(self):
        """Test production environment settings."""
        settings = ProductionSettings()
        
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.host == "0.0.0.0"
        assert settings.workers == 4
    
    def test_testing_settings(self):
        """Test testing environment settings."""
        settings = UnitTestSettings()
        
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.openai_api_key == "test-key"
        assert settings.host == "127.0.0.1"
        assert settings.port == 8001
    
    def test_get_environment_settings_development(self):
        """Test getting development settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            settings = get_environment_settings()
            assert isinstance(settings, DevelopmentSettings)
            assert settings.debug is True
    
    def test_get_environment_settings_production(self):
        """Test getting production settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = get_environment_settings()
            assert isinstance(settings, ProductionSettings)
            assert settings.debug is False
    
    def test_get_environment_settings_testing(self):
        """Test getting testing settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'testing'}):
            settings = get_environment_settings()
            assert isinstance(settings, UnitTestSettings)
            assert settings.openai_api_key == "test-key"
    
    def test_get_environment_settings_default(self):
        """Test getting default (development) settings."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_environment_settings()
            assert isinstance(settings, DevelopmentSettings)
    
    def test_get_environment_settings_unknown(self):
        """Test getting settings for unknown environment."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'unknown'}):
            settings = get_environment_settings()
            assert isinstance(settings, DevelopmentSettings)  # Should default to development


class TestSettingsCaching:
    """Test cases for settings caching."""
    
    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Should be the same instance due to lru_cache
        assert settings1 is settings2