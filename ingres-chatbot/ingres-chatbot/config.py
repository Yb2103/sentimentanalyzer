"""
INGRES ChatBot Configuration
Centralized configuration management for the INGRES AI-driven ChatBot system
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Settings
    APP_NAME: str = "INGRES ChatBot"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    TESTING: bool = False
    
    # Server Configuration
    HOST: str = "AIChatbot"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    INGRES_DB_HOST: str = "localhost"
    INGRES_DB_PORT: int = 5432
    INGRES_DB_NAME: str = "ingres"
    INGRES_DB_USER: str = "ingres_user"
    INGRES_DB_PASSWORD: str = ""
    
    # Redis Configuration (for caching and sessions)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TIMEOUT: int = 3600  # 1 hour
    
    # AI/ML Configuration
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CONTEXT_LENGTH: int = 4096
    
    # NLP Configuration
    SPACY_MODEL: str = "en_core_web_sm"
    SUPPORTED_LANGUAGES: List[str] = [
        "en", "hi", "mr", "gu", "bn", "ta", "te", "kn", "ml", "pa"
    ]
    DEFAULT_LANGUAGE: str = "en"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "your-jwt-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File Upload Settings
    UPLOAD_FOLDER: str = "uploads"
    MAX_FILE_SIZE: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".json", ".txt"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/chatbot.log"
    
    # External API Settings
    INGRES_API_URL: str = "https://ingres.iith.ac.in/api"
    GOOGLE_TRANSLATE_API_KEY: Optional[str] = None
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 30
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @validator('DATABASE_URL', pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str):
            return v
        return (
            f"postgresql://{values.get('INGRES_DB_USER')}:"
            f"{values.get('INGRES_DB_PASSWORD')}@"
            f"{values.get('INGRES_DB_HOST')}:"
            f"{values.get('INGRES_DB_PORT')}/"
            f"{values.get('INGRES_DB_NAME')}"
        )


# Initialize settings
settings = Settings()


class GroundwaterDataConfig:
    """Configuration for groundwater data processing and queries"""
    
    # Assessment Categories
    CATEGORIES = {
        "safe": "Safe",
        "semi_critical": "Semi-Critical", 
        "critical": "Critical",
        "over_exploited": "Over-Exploited"
    }
    
    # Assessment Parameters
    PARAMETERS = {
        "annual_extractable_groundwater_resource": "Annual Extractable Groundwater Resource",
        "annual_groundwater_extraction": "Annual Groundwater Extraction",
        "stage_of_groundwater_extraction": "Stage of Groundwater Extraction",
        "net_annual_groundwater_availability": "Net Annual Groundwater Availability",
        "annual_groundwater_recharge": "Annual Groundwater Recharge"
    }
    
    # Administrative Levels
    ADMIN_LEVELS = ["state", "district", "block", "mandal", "taluk"]
    
    # Data Validation Rules
    VALIDATION_RULES = {
        "stage_of_extraction": {"min": 0, "max": 200},
        "groundwater_extraction": {"min": 0},
        "groundwater_recharge": {"min": 0}
    }


class IntentConfig:
    """Configuration for NLP intent recognition"""
    
    # Intent Categories
    INTENTS = {
        "groundwater_status": {
            "patterns": [
                "what is the groundwater status",
                "show groundwater level",
                "groundwater condition",
                "water table status"
            ],
            "response_type": "data_query"
        },
        "historical_data": {
            "patterns": [
                "show trend",
                "historical data",
                "past years",
                "compare with previous"
            ],
            "response_type": "trend_analysis"
        },
        "comparison": {
            "patterns": [
                "compare with",
                "versus",
                "difference between",
                "how does it compare"
            ],
            "response_type": "comparative_analysis"
        },
        "geographical_query": {
            "patterns": [
                "in maharashtra",
                "karnataka groundwater",
                "show state wise",
                "district data"
            ],
            "response_type": "location_query"
        },
        "export_data": {
            "patterns": [
                "download",
                "export",
                "save data",
                "generate report"
            ],
            "response_type": "data_export"
        }
    }
    
    # Entity Types
    ENTITIES = {
        "LOCATION": ["STATE", "DISTRICT", "BLOCK", "MANDAL", "TALUK"],
        "PARAMETER": ["RECHARGE", "EXTRACTION", "STAGE", "AVAILABILITY"],
        "TIME": ["YEAR", "MONTH", "SEASON", "PERIOD"],
        "CATEGORY": ["SAFE", "SEMI_CRITICAL", "CRITICAL", "OVER_EXPLOITED"]
    }


class VisualizationConfig:
    """Configuration for data visualization and charts"""
    
    # Chart Types
    CHART_TYPES = {
        "bar_chart": "Bar Chart",
        "line_chart": "Line Chart", 
        "pie_chart": "Pie Chart",
        "map": "Geographic Map",
        "scatter_plot": "Scatter Plot",
        "heatmap": "Heatmap"
    }
    
    # Color Schemes
    COLOR_SCHEMES = {
        "groundwater_status": {
            "safe": "#2E8B57",
            "semi_critical": "#FFD700", 
            "critical": "#FF8C00",
            "over_exploited": "#DC143C"
        },
        "trend_analysis": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        "default": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    }
    
    # Map Configuration
    MAP_CONFIG = {
        "center_lat": 20.5937,
        "center_lon": 78.9629,
        "zoom_level": 5,
        "tile_layer": "OpenStreetMap"
    }


class ResponseTemplates:
    """Response templates for different query types"""
    
    GREETING = [
        "Hello! I'm your INGRES assistant. I can help you with groundwater data queries.",
        "Welcome to INGRES ChatBot! How can I help you with groundwater information today?",
        "Hi there! I'm here to assist you with India's groundwater resource data."
    ]
    
    DATA_NOT_FOUND = [
        "I couldn't find data for your query. Please check the location name or try a different search.",
        "No data available for the specified criteria. Would you like to try a different query?",
        "Sorry, I don't have information matching your request. Can you be more specific?"
    ]
    
    ERROR_MESSAGES = {
        "invalid_location": "The location you specified is not recognized. Please provide a valid state, district, or block name.",
        "invalid_year": "Please provide a valid year between 2005 and 2023.",
        "database_error": "I'm experiencing some technical difficulties. Please try again in a moment.",
        "rate_limit": "Too many requests. Please wait a moment before trying again."
    }
    
    HELP_MESSAGES = {
        "examples": [
            "Try: 'What is the groundwater status in Maharashtra?'",
            "Try: 'Show me the trend for Karnataka over the last 5 years'", 
            "Try: 'Compare groundwater levels between Punjab and Haryana'",
            "Try: 'Which districts in Rajasthan are over-exploited?'"
        ]
    }


# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "DEBUG": True,
    "LOG_LEVEL": "DEBUG",
    "CACHE_TIMEOUT": 300,
    "RATE_LIMIT_PER_MINUTE": 100
}

PRODUCTION_CONFIG = {
    "DEBUG": False,
    "LOG_LEVEL": "WARNING", 
    "CACHE_TIMEOUT": 3600,
    "RATE_LIMIT_PER_MINUTE": 30,
    "WORKERS": 8
}

TESTING_CONFIG = {
    "TESTING": True,
    "DATABASE_URL": "sqlite:///test.db",
    "CACHE_TIMEOUT": 60
}

# Export main settings instance
__all__ = [
    "settings", 
    "GroundwaterDataConfig", 
    "IntentConfig", 
    "VisualizationConfig", 
    "ResponseTemplates"
]