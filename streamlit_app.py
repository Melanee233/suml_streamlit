import streamlit as st
import pandas as pd
import time
import matplotlib as plt
import os
from transformers import MarianTokenizer, MarianMTModel


# Konfiguracja strony
st.set_page_config(
    page_title="Tłumacz EN → DE",
    page_icon="🌍",
    layout="wide"
)

# Tytuł i emoji
st.markdown("# 🌍 Translator EN ↔ DE, autor: s27735")
st.markdown("### Tłumacz z angielskiego na niemiecki")

# Informacja o aplikacji
with st.expander("ℹ️ O aplikacji"):
    st.markdown("""
    **Witaj w aplikacji tłumaczącej!**
    
    Ta aplikacja pozwala na szybkie i łatwe tłumaczenie tekstu:
    - 🇬🇧 **Z angielskiego** na 🇩🇪 **niemiecki**
    - Wykorzystuje model AI oparty na architekturze **Marian** (Helsinki-NLP/opus-mt-en-de)
    - Obsługuje zarówno krótkie zdania jak i dłuższe fragmenty tekstu
    
    **Jak z niej korzystać?**
    1. Wpisz tekst w angielskim w pole poniżej
    2. Naciśnij Enter lub kliknij poza polem
    3. Czekaj na przetłumaczenie
    4. Wynik pojawi się poniżej! ✨
    """)

st.markdown("---")

# Sekcja główna
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📝 Wpisz tekst do tłumaczenia")
    text = st.text_area(
        label="Angielski tekst:",
        placeholder="np. Hello, how are you today?",
        height=150,
        label_visibility="collapsed"
    )

with col2:
    st.subheader("🎯 Wynik tłumaczenia")
    result_placeholder = st.empty()

# Tłumaczenie
if text:
    try:
        with st.spinner('⏳ Ładuję model i tłumaczę tekst...'):
            tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model.generate(**inputs)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        with result_placeholder.container():
            st.success("✅ Tłumaczenie gotowe!")
            st.text_area(
                label="Niemiecki tekst:",
                value=answer,
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
            st.info(f"📊 Długość oryginalnego tekstu: {len(text)} znaków")
            
    except Exception as e:
        result_placeholder.error(f"❌ Błąd podczas tłumaczenia: {str(e)}")
        st.warning("Spróbuj ponownie lub sprawdź połączenie internetowe.")
else:
    result_placeholder.info("👆 Wpisz tekst w polu obok, aby rozpocząć tłumaczenie")

