"""Utility helpers for generating bcrypt hashes compatible with the app.

Este script produce hashes bcrypt (sin dependencias externas) para que puedas
copiarlos y pegarlos en ``config_users.yaml`` u otro archivo de configuración.

Uso rápido:
    $ python tools/hash_streamlit_passwords.py --interactive

También puedes pasar las contraseñas como argumentos (solo para pruebas locales):
    $ python tools/hash_streamlit_passwords.py "MiClaveSecreta" "OtraClave2025!"

Recuerda que escribir contraseñas en la línea de comandos puede dejarlas en el
historial de tu shell. El modo ``--interactive`` evita eso porque usa ``getpass``.
"""

from __future__ import annotations

import argparse
import getpass
from typing import Iterable, List

import bcrypt


def generate_hashes(passwords: Iterable[str]) -> List[str]:
    """Return the bcrypt hashes for the provided passwords."""

    password_list = list(passwords)
    if not password_list:
        return []

    hashes: List[str] = []
    for password in password_list:
        if not password:
            continue
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
        hashes.append(hashed.decode("utf-8"))
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera hashes bcrypt compatibles con la app Streamlit",
    )
    parser.add_argument(
        "passwords",
        nargs="*",
        help=(
            "Contraseñas en texto plano. Usa solo para pruebas locales; para mayor "
            "seguridad ejecuta el script con --interactive y sin argumentos."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Activa un flujo interactivo para ingresar contraseñas con getpass "
            "(no quedan en el historial de la terminal)."
        ),
    )
    return parser.parse_args()


def interactive_passwords() -> Iterable[str]:
    """Yield passwords introducidas de forma interactiva."""

    print("Introduce cada contraseña cuando se te solicite. Deja vacío y pulsa Enter para terminar.\n")
    while True:
        password = getpass.getpass("Contraseña: ")
        if not password:
            break
        confirmation = getpass.getpass("Repite la contraseña para confirmar: ")
        if password != confirmation:
            print("Las contraseñas no coinciden. Intenta de nuevo.\n")
            continue
        yield password


def main() -> None:
    args = parse_args()

    if args.interactive:
        passwords = list(interactive_passwords())
    else:
        passwords = args.passwords

    hashes = generate_hashes(passwords)

    if not hashes:
        print("No se proporcionaron contraseñas. Nada que hashear.")
        return

    print("\nHashes generados (copia y pega cada cadena completa en tu YAML):\n")
    for idx, hash_value in enumerate(hashes, start=1):
        print(f"{idx}. {hash_value}")


if __name__ == "__main__":
    main()
