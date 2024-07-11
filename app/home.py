import base64
from io import BytesIO
import requests
import streamlit as st

def page_home():
    st.title("🎹 piano_to_note 🎹")
    st.markdown(
        "### Convertissez votre fichier MP3 en partition en quelques clics ! 🎶\n"
        "Téléchargez un fichier MP3 ci-dessous pour commencer."
    )
    st.markdown(
        """
        **piano_to_note** est une application révolutionnaire conçue pour les musiciens, les compositeurs et les passionnés de musique. 
        Son objectif est de simplifier le processus de transcription de la musique en partition. 
        Voici comment cette application peut vous être utile :

        - **Facilité d'utilisation** : Téléchargez simplement votre fichier MP3 et laissez l'application faire le reste.
        - **Gain de temps** : Obtenez rapidement une partition à partir de votre fichier audio sans avoir besoin de transcrire manuellement.
        - **Précision** : Bénéficiez d'une transcription précise grâce à des algorithmes avancés de traitement du son.
        - **Visualisation** : Visualisez l'onde sonore de votre fichier pour mieux comprendre la structure musicale.
        """
    )

    uploaded_file = st.file_uploader("Choisir un fichier MP3", type=["mp3"])

    if uploaded_file is not None:
        st.success(f"Vous avez téléchargé : {uploaded_file.name}")

        if st.button("Convertir en partition"):
            st.info("Traitement en cours...")

            with st.spinner('Conversion en cours...'):
                # Convert MP3 to MIDI
                midi_response = requests.post(
                    "http://api:8000/convert_file_to_midi",
                    files={"file": uploaded_file.getvalue()}
                )

                if midi_response.status_code == 200:
                    midi_file = BytesIO(midi_response.content)
                    st.session_state['midi_file'] = midi_file
                    
                    # Convert MIDI to PDF
                    pdf_response = requests.post(
                        "http://api:8000/convert-midi-to-pdf/",
                        files={"file": midi_file}
                    )

                    if pdf_response.status_code == 200:
                        pdf_file = BytesIO(pdf_response.content)
                        st.session_state['pdf_file'] = pdf_file
                        st.success("Conversion terminée ! 🎵")

                        # Display the PDF
                        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)

                        pdf_file.seek(0)
                        # Provide a download button
                        st.download_button(
                            label="Télécharger la partition PDF",
                            data=pdf_file,
                            file_name="partition.pdf",
                            mime="application/pdf"
                        )
                        
                        
                        midi_file.seek(0)
                        st.download_button(
                            label="Télécharger le fichier MIDI",
                            data=midi_file,
                            file_name="transcribed.mid",
                            mime="audio/midi"
                        )
                    else:
                        st.error("Erreur lors de la conversion du fichier MIDI en PDF")
                else:
                    st.error("Erreur lors de la conversion du fichier MP3 en MIDI")

    st.markdown("---")
    st.markdown("Créé avec la joie de vivre ❤️ par les boss du troll alias JP & Ali")
