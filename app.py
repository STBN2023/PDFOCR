#!/usr/bin/env python3
"""
PDF Text Extraction Tool
-----------------------
Application Streamlit pour l'extraction de texte depuis des PDFs
avec OCR et transformation en CSV.
Version améliorée avec support multi-pages, options configurables,
et gestion des modèles d'export.
"""

import streamlit as st
import io
import re
import cv2
import numpy as np
import pandas as pd
import pytesseract
import csv
import json
import logging
from PIL import Image
from pdf2image import convert_from_bytes
from streamlit_cropper import st_cropper
from config import APP_CONFIG, TESSERACT_CONFIG, TESSERACT_LANG, init_app
from typing import List, Dict, Any, Optional, Tuple, Union

# Configuration du logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialisation de l'application
init_app()

def save_export_model(model_name: str, selected_columns: list):
    """
    Sauvegarde le modèle d'export (liste des colonnes) dans un fichier JSON.
    """
    models_file = "export_models.json"
    try:
        with open(models_file, "r", encoding="utf-8") as f:
            models = json.load(f)
    except Exception:
        models = {}
    models[model_name] = selected_columns
    with open(models_file, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=4, ensure_ascii=False)

def load_export_models() -> dict:
    """
    Charge les modèles d'export depuis le fichier JSON.
    """
    models_file = "export_models.json"
    try:
        with open(models_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def merge_ocr_lines(extracted_text):
    """
    Fusionne les lignes issues de l'OCR pour reconstituer chaque enregistrement.
    """
    lines = extracted_text.splitlines()
    merged_records = []
    current_record = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d{6}$', line):
            if current_record:
                merged_records.append(current_record)
            current_record = line
        else:
            current_record += " " + line
    if current_record:
        merged_records.append(current_record)
    return merged_records

def process_record_line(line):
    """
    Traite une chaîne de caractères correspondant à un enregistrement OCR complet.
    """
    line = re.sub(r'\s*@\s*', '@', line)
    line = re.sub(r'\s*\.\s*', '.', line)
    
    m = re.match(r'^\s*(\d{6})', line)
    num_appart = m.group(1) if m else ""
    
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', line)
    phones = re.findall(r'\d[\d\s\.\-]{5,}\d', line)
    
    line_clean = line
    for e in emails:
        line_clean = line_clean.replace(e, '')
    for p in phones:
        line_clean = line_clean.replace(p, '')
    line_clean = re.sub(r'\s*\([^)]*\)', '', line_clean)
    
    tokens = line_clean.split()
    if tokens and tokens[0] == num_appart:
        tokens = tokens[1:]
    
    num_residence = ""
    split_index = None
    for i, t in enumerate(tokens):
        if t.isdigit() and len(t) == 4:
            num_residence = t
            split_index = i
            break
            
    if split_index is not None:
        name = " ".join(tokens[:split_index])
        nom_residence = " ".join(tokens[split_index+1:])
    else:
        name = " ".join(tokens)
        nom_residence = ""
    
    emails = [e.strip() for e in emails]
    while len(emails) < 3:
        emails.append("")
    emails = emails[:3]
    
    phones = [p.strip() for p in phones]
    while len(phones) < 2:
        phones.append("")
    phones = phones[:2]
    
    return {
        "NUM_APPART": num_appart,
        "NAME": name.strip(),
        "NUM_RESIDENCE": num_residence.strip(),
        "NOM_RESIDENCE": nom_residence.strip(),
        "EMAIL 1": emails[0],
        "EMAIL 2": emails[1],
        "EMAIL 3": emails[2],
        "TEL 1": phones[0],
        "TEL 2": phones[1]
    }

def apply_image_enhancements(image: np.ndarray, brightness: float = 1.0,
                             contrast: float = 1.0, threshold_method: str = 'otsu') -> np.ndarray:
    """
    Applique des améliorations configurables à l'image.
    """
    try:
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        if threshold_method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        logger.info(f"Amélioration de l'image appliquée: {threshold_method}")
        return thresh
    except Exception as e:
        logger.error(f"Erreur lors de l'amélioration de l'image: {str(e)}")
        raise

def process_pdf_pages(pdf_bytes: bytes, dpi: int = 300) -> List[Image.Image]:
    """
    Convertit toutes les pages d'un PDF en images.
    """
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        logger.info(f"Conversion PDF réussie: {len(pages)} pages")
        return pages
    except Exception as e:
        logger.error(f"Erreur lors de la conversion du PDF: {str(e)}")
        raise

def batch_process_images(images: List[Image.Image], options: Dict[str, Any]) -> pd.DataFrame:
    """
    Traite un lot d'images et combine les résultats.
    """
    all_records = []
    for idx, img in enumerate(images):
        try:
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed = apply_image_enhancements(
                cv_image,
                brightness=options['brightness'],
                contrast=options['contrast'],
                threshold_method=options['threshold_method']
            )
            
            text = pytesseract.image_to_string(
                processed,
                config=f"{TESSERACT_CONFIG} {options['extra_config']}",
                lang=TESSERACT_LANG
            )
            
            merged_records = merge_ocr_lines(text)
            processed_records = [process_record_line(rec) for rec in merged_records if rec.strip()]
            
            for record in processed_records:
                record['PAGE'] = idx + 1
            
            all_records.extend(processed_records)
            logger.info(f"Page {idx + 1} traitée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la page {idx + 1}: {str(e)}")
            st.error(f"Erreur sur la page {idx + 1}: {str(e)}")
    
    return pd.DataFrame(all_records) if all_records else None

def display_sidebar_options() -> Dict[str, Any]:
    """
    Affiche et gère les options de configuration dans la barre latérale.
    """
    with st.sidebar:
        st.header("Options de traitement")
        
        # Options PDF
        st.subheader("Options PDF")
        dpi = st.slider("DPI", 150, 600, APP_CONFIG['PDF_DPI'])
        process_all_pages = st.checkbox("Traiter toutes les pages", value=False)
        
        # Options d'image
        st.subheader("Amélioration d'image")
        brightness = st.slider("Luminosité", 0.5, 2.0, 1.0)
        contrast = st.slider("Contraste", 0.5, 2.0, 1.0)
        threshold_method = st.selectbox(
            "Méthode de seuillage",
            ['otsu', 'adaptive', 'simple'],
            index=0
        )
        
        # Options OCR
        st.subheader("Options OCR")
        extra_config = st.text_input("Configuration Tesseract additionnelle", "")
        
        # Options d'export CSV
        if st.session_state.get('processed_data') is not None:
            st.subheader("Options d'export")
            all_columns = APP_CONFIG['all_columns']
            
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = all_columns.copy()
            
            selected_columns = st.multiselect(
                "Colonnes à exporter",
                all_columns,
                default=st.session_state.selected_columns
            )
            st.session_state.selected_columns = selected_columns
            
            st.markdown("### Sauvegarder le modèle d'export")
            model_name = st.text_input("Nom du modèle", key="model_name")
            if st.button("Enregistrer le modèle d'export"):
                if model_name:
                    save_export_model(model_name, selected_columns)
                    st.success(f"Modèle '{model_name}' sauvegardé avec succès.")
                else:
                    st.error("Veuillez entrer un nom pour le modèle.")
            
            # Chargement d'un modèle sauvegardé
            saved_models = load_export_models()
            if saved_models:
                st.markdown("### Charger un modèle d'export")
                # Option 1 : Charger le dernier modèle utilisé
                if st.checkbox("Charger le dernier modèle utilisé", key="load_last_model"):
                    last_model = list(saved_models.keys())[-1]
                    st.session_state.selected_columns = saved_models[last_model]
                    st.info(f"Modèle '{last_model}' chargé.")
                
                # Option 2 : Choisir parmi une liste de modèles sauvegardés
                model_list = list(saved_models.keys())
                chosen_model = st.selectbox("Ou choisissez un modèle", model_list, key="chosen_model")
                if st.button("Charger le modèle sélectionné", key="load_selected_model"):
                    st.session_state.selected_columns = saved_models[chosen_model]
                    st.success(f"Modèle '{chosen_model}' chargé avec succès.")
        
        return {
            'dpi': dpi,
            'process_all_pages': process_all_pages,
            'brightness': brightness,
            'contrast': contrast,
            'threshold_method': threshold_method,
            'extra_config': extra_config,
            'selected_columns': st.session_state.get('selected_columns', APP_CONFIG['all_columns'])
        }

def export_to_csv(df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[Union[bytes, None], bool]:
    """
    Prépare et retourne les données CSV avec les colonnes sélectionnées, en utilisant la virgule comme séparateur.
    """
    selected_columns = options.get('selected_columns', [])
    if not selected_columns:
        st.warning("Veuillez sélectionner au moins une colonne à exporter")
        return None, False

    try:
        # Filtrer les colonnes existantes
        available_columns = [col for col in selected_columns if col in df.columns]
        if not available_columns:
            st.warning("Aucune colonne sélectionnée n'est disponible")
            return None, False

        # Créer le CSV avec les colonnes sélectionnées et la virgule comme séparateur
        csv_data = df[available_columns].to_csv(
            index=False,
            sep=',',
            quoting=csv.QUOTE_ALL,
            encoding=APP_CONFIG['CSV_ENCODING']
        ).encode(APP_CONFIG['CSV_ENCODING'])

        return csv_data, True

    except Exception as e:
        logger.error(f"Erreur lors de l'export CSV : {str(e)}")
        st.error("Une erreur s'est produite lors de l'export")
        return None, False

def show_editable_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche un DataFrame éditable avec navigation entre les enregistrements.
    """
    st.subheader("Édition des résultats")
    st.write("Modifiez les valeurs si nécessaire avant l'export :")

    if 'edited_df' not in st.session_state:
        st.session_state.edited_df = df.copy()
    
    edited_df = st.session_state.edited_df

    if len(edited_df) == 0:
        st.warning("Aucun enregistrement à éditer")
        return edited_df

    total_records = len(edited_df)
    
    if st.session_state.current_record_index >= total_records:
        st.session_state.current_record_index = total_records - 1
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("← Précédent", disabled=st.session_state.current_record_index == 0):
            st.session_state.current_record_index = max(0, st.session_state.current_record_index - 1)
            
    with col2:
        st.write(f"Enregistrement {st.session_state.current_record_index + 1} sur {total_records}")
        
    with col3:
        if st.button("Suivant →", disabled=st.session_state.current_record_index == total_records - 1):
            st.session_state.current_record_index = min(total_records - 1, st.session_state.current_record_index + 1)
    
    current_idx = st.session_state.current_record_index
    row = edited_df.iloc[current_idx]
    row_idx = edited_df.index[current_idx]
    
    st.markdown(f"**Numéro d'appartement: {row['NUM_APPART']}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        edited_df.at[row_idx, 'NUM_APPART'] = st.text_input(
            "Numéro d'appartement",
            value=row['NUM_APPART'],
            key=f"num_appart_{current_idx}"
        )
        edited_df.at[row_idx, 'NAME'] = st.text_input(
            "Nom",
            value=row['NAME'],
            key=f"name_{current_idx}"
        )
        edited_df.at[row_idx, 'NUM_RESIDENCE'] = st.text_input(
            "Numéro de résidence",
            value=row['NUM_RESIDENCE'],
            key=f"num_res_{current_idx}"
        )
        edited_df.at[row_idx, 'NOM_RESIDENCE'] = st.text_input(
            "Nom de résidence",
            value=row['NOM_RESIDENCE'],
            key=f"nom_res_{current_idx}"
        )
    
    with col2:
        edited_df.at[row_idx, 'EMAIL 1'] = st.text_input(
            "Email 1",
            value=row['EMAIL 1'],
            key=f"email1_{current_idx}"
        )
        edited_df.at[row_idx, 'EMAIL 2'] = st.text_input(
            "Email 2",
            value=row['EMAIL 2'],
            key=f"email2_{current_idx}"
        )
        edited_df.at[row_idx, 'EMAIL 3'] = st.text_input(
            "Email 3",
            value=row['EMAIL 3'],
            key=f"email3_{current_idx}"
        )
        edited_df.at[row_idx, 'TEL 1'] = st.text_input(
            "Téléphone 1",
            value=row['TEL 1'],
            key=f"tel1_{current_idx}"
        )
        edited_df.at[row_idx, 'TEL 2'] = st.text_input(
            "Téléphone 2",
            value=row['TEL 2'],
            key=f"tel2_{current_idx}"
        )

    if st.checkbox("Supprimer cet enregistrement", key=f"delete_{current_idx}"):
        edited_df.drop(row_idx, inplace=True)
        st.warning("Cet enregistrement sera supprimé lors de l'export")
        if current_idx >= len(edited_df):
            st.session_state.current_record_index = max(0, len(edited_df) - 1)
        st.rerun()

    st.subheader("Aperçu des données modifiées")
    st.dataframe(edited_df)
    
    st.session_state.edited_df = edited_df
    
    return edited_df

def main():
    st.title("OCR, Recadrage & Transformation CSV")
    st.markdown("""
    Version améliorée avec :
    - Support multi-pages pour les PDF
    - Options d'amélioration d'image configurables
    - Traitement par lot
    - Configuration OCR flexible
    - Édition manuelle des résultats
    - Sauvegarde et chargement des modèles d'export
    """)
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'current_record_index' not in st.session_state:
        st.session_state.current_record_index = 0
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    options = display_sidebar_options()
    
    uploaded_file = st.file_uploader(
        "Uploader une image ou un PDF",
        type=APP_CONFIG['ALLOWED_EXTENSIONS']
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                pages = process_pdf_pages(uploaded_file.read(), options['dpi'])
                if len(pages) > 1 and not options['process_all_pages']:
                    page_number = st.slider("Sélectionner la page", 1, len(pages), 1)
                    pages = [pages[page_number - 1]]
            else:
                pages = [Image.open(uploaded_file)]
            
            processed_pages = []
            
            for idx, page in enumerate(pages):
                st.subheader(f"Page {idx + 1}")
                st.image(page, use_container_width=True)
                
                st.write("Sélectionnez la zone à traiter :")
                cropped_page = st_cropper(page, realtime_update=True, aspect_ratio=None)
                st.image(cropped_page, use_container_width=True, caption="Zone sélectionnée")
                
                processed_pages.append(cropped_page)
            
            if not st.session_state.processing_complete:
                if st.button("Traiter les zones sélectionnées"):
                    with st.spinner("Traitement en cours..."):
                        df = batch_process_images(processed_pages, options)
                        if df is not None:
                            st.session_state.processed_data = df
                            st.session_state.processing_complete = True
                            st.rerun()
            
            if st.session_state.processing_complete and st.session_state.processed_data is not None:
                df = st.session_state.processed_data
                st.success(f"Traitement terminé : {len(df)} enregistrements trouvés")
                
                edited_df = show_editable_dataframe(df)
                
                if st.button("Exporter en CSV"):
                    csv_data, is_valid = export_to_csv(edited_df, options)
                    if is_valid:
                        st.download_button(
                            label="Télécharger le CSV corrigé",
                            data=csv_data,
                            file_name="donnees_transformees.csv",
                            mime="text/csv"
                        )
                
                if st.button("Nouveau traitement"):
                    st.session_state.processed_data = None
                    st.session_state.processing_complete = False
                    st.session_state.current_record_index = 0
                    st.rerun()
        
        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")
            logger.error(f"Erreur générale : {str(e)}")

if __name__ == "__main__":
    main()
