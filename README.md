# podcast-voice-pipeline

Pipeline open source para:

- transcripción de audio
- diarización de hablantes
- generación de `.srt` completo
- generación de `.srt` por hablante
- separación de audio por hablante
- fichas de voz para clonación posterior

## Estructura de entrada

Cada audio vive en su propio directorio:

inputs/<audio_id>/<audio_id>.mp3

Ejemplo:

inputs/test001/test001.mp3

## Salidas

Se generan dentro de:

inputs/<audio_id>/outputs/

## Secrets necesarios

En GitHub repo settings > Secrets and variables > Actions:

- `HF_TOKEN`: token de Hugging Face con acceso al modelo de pyannote

## Workflow

Ejecuta el workflow manualmente indicando:

- `audio_id`: por ejemplo `test001`

## Resultado esperado

inputs/test001/outputs/
- full/transcript.json
- full/full.srt
- by_speaker/SPEAKER_00.srt
- by_speaker/SPEAKER_01.srt
- voices/SPEAKER_00.md
- voices/SPEAKER_00.json
- audio_by_speaker/SPEAKER_00/*.wav
- manifests/segments_by_speaker.json
