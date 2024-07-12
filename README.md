## Piano-To-Notes

This project aims to transcribe notes into score.

### Technologies Used

- Python
- Tensorflow
- Streamlit
- Librosa
- FastAPI
- PostgreSQL
- Docker

### Installation Steps

To set up and run the application using Docker, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/JeanPhilippeCaetano/piano-to-notes.git
   cd piano-to-notes/
   ```

2. Build and run the Docker containers:
   ```sh
   docker-compose up --build
   ```

### API Routes

| Route                   | Method | Description                               |
|-------------------------|--------|-------------------------------------------|
| `/convert_file_to_midi` | POST   | Convert an audio file to a MIDI file      |
| `/convert_midi_to_pdf`  | POST   | Convert a MIDI file to a sheet music PDF  |

### Example Usage (with Linux or Mac OS)

#### Convert an Audio File to a MIDI File

```sh
curl -X POST http://localhost:5000/convert_file_to_midi -H "Content-Type: multipart/form-data" -F 'file=@/path/to/audiofile.wav'
```

#### Convert a MIDI File to a Sheet Music PDF

```sh
curl -X POST http://localhost:5000/convert_midi_to_pdf -H "Content-Type: multipart/form-data" -F 'file=@/path/to/midifile.mid'
```

### Notes

- Ensure Docker and Docker Compose are installed on your machine.
- Update the `compose.yml` and database configuration in `config.py` as needed.
