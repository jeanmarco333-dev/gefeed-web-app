
# JM P-Feedlot v0.26 — 100% Web (Streamlit)

## Local
pip install -r requirements.txt
streamlit run app.py

## Cloud (gratis)
- GitHub repo con estos archivos
- https://share.streamlit.io → New app → seleccionar repo y `app.py`

## Configurar usuarios (autenticación)

1. Sigue la guía detallada en [docs/paso_a_paso_autenticacion.md](docs/paso_a_paso_autenticacion.md).
2. Usa el script `python tools/hash_streamlit_passwords.py --interactive` para generar hashes seguros.
3. Copia los hashes en `config_users.yaml`, actualiza las cookies y reinicia la app.
