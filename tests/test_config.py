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
    
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'PORT': '9000',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'LOG_LEVEL': 'DEBUG'
        }):
            settings = Settings()
            
            assert settings.debug is True
            assert settings.port == 9000
            assert settings.openai_model == "gpt-3.5-turbo"
            assert settings.log_level == "DEBUG"
    
    def test_temperature_validation_valid(self):
        """Test valid temperature values."""
        # Should not raise exception
        Settings(openai_temperature=0.0)
        Settings(openai_temperature=1.0)
        Settings(openai_temperature=2.0)
    
    def test_temperature_validation_invalid(self):
        """Test invalid temperature values."""
        with pytest.raises(ValueError, match="Temperature must be between"):
            Settings(openai_temperature=-0.1)
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            Settings(openai_temperature=2.1)
    
    def test_log_level_validation_valid(self):
        """Test valid log level values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level
        
        # Test case insensitive
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"
    
    def test_log_level_validation_invalid(self):
        """Test invalid log level values."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            Settings(log_level="INVALID")
    
    def test_cors_origins_parsing_string(self):
        """Test parsing CORS origins from string."""
        with patch.dict(os.environ, {'CORS_ORIGINS': 'http://localhost:3000,https://example.com'}):
            settings = Settings()
            expected = ["http://localhost:3000", "https://example.com"]
            assert settings.cors_origins == expected
    
    def test_cors_origins_parsing_list(self):
        """Test CORS origins as list."""
        origins = ["http://localhost:3000", "https://example.com"]
        settings = Settings(cors_origins=origins)
        assert settings.cors_origins == origins
    
    def test_ticket_categories_parsing_string(self):
        """Test parsing ticket categories from string."""
        categories_str = "technical,billing,general"
        settings = Settings(ticket_categories=categories_str)
        expected = ["technical", "billing", "general"]
        assert settings.ticket_categories == expected
    
    def test_ticket_categories_parsing_list(self):
        """Test ticket categories as list."""
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