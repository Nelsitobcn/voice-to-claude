"""Quita las frases que Whisper se inventa cuando oye silencio.

Medido el 20-nov-2026 en ~/Library/Logs/voice-to-claude.daemon.log: con Nelson callado,
salían grabaciones enteras transcritas como «Gracias por ver el video.» (16 veces) y
Whisper también las pega al final de un dictado real. Son frases de subtítulos de YouTube
con las que se entrenó el modelo, no algo que Nelson diga.
"""
import re
import unicodedata

# Frases que solo salen de vídeos de YouTube: se quitan siempre, solas o al final.
FANTASMA = [
    "gracias por ver el video",
    "gracias por ver",
    "subtitulos realizados por la comunidad de amara.org",
    "subtitulos por la comunidad de amara.org",
    "suscribete al canal",
    "no olvides suscribirte",
]


def _normal(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9. ]+", " ", t).strip(" .")


def limpiar(texto: str) -> str:
    """Devuelve el texto sin frases fantasma; cadena vacía si solo era eso."""
    if not texto:
        return texto
    trozos = re.split(r"(?<=[.!?…])\s+", texto.strip())
    while trozos and any(_normal(trozos[-1]).strip() == f for f in FANTASMA):
        trozos.pop()
    while trozos and any(_normal(trozos[0]).strip() == f for f in FANTASMA):
        trozos.pop(0)
    return " ".join(trozos).strip()
