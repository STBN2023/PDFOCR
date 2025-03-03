"""
Configuration for OCR PDF Text Extraction Tool
--------------------------------------------
Author: Stephane
Version: 1.0
"""

import sys
from typing import Dict, Any, List
import importlib
from PIL import Image

# Configuration Tesseract
TESSERACT_CONFIG = '--psm 12'
TESSERACT_LANG = 'fra'

# Configuration de l'application
APP_CONFIG: Dict[str, Any] = {
    'ALLOWED_EXTENSIONS': ['png', 'jpg', 'jpeg', 'pdf'],
    'PDF_DPI': 300,
    'CSV_SEPARATOR': ';',
    'CSV_ENCODING': 'utf-8',
    'all_columns': [
        "NUM_APPART", 
        "NAME", 
        "NUM_RESIDENCE", 
        "NOM_RESIDENCE",
        "EMAIL 1", 
        "EMAIL 2", 
        "EMAIL 3", 
        "TEL 1", 
        "TEL 2", 
        "PAGE"
    ]
}

def verify_package(package_name: str, import_name: str) -> bool:
    """Vérifie si un package est correctement installé."""
    try:
        module = importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def check_dependencies() -> bool:
    """Vérifie que toutes les dépendances requises sont installées."""
    required_packages = {
        'streamlit': 'streamlit',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'pytesseract': 'pytesseract',
        'Pillow': 'PIL',
        'pdf2image': 'pdf2image',
        'streamlit_cropper': 'streamlit_cropper'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        if not verify_package(package_name, import_name.split('.')[0]):
            missing_packages.append(package_name)
    
    if missing_packages:
        missing_str = ', '.join(missing_packages)
        print(f"Packages manquants : {missing_str}")
        print("Installez-les avec : pip install -r requirements.txt")
        return False

    return True

def init_app() -> bool:
    """
    Initialise l'application et vérifie les prérequis.
    
    Returns:
        bool: True si l'initialisation est réussie, False sinon
    """
    # Vérification des dépendances
    if not check_dependencies():
        return False
    
    return True