# Autenticación en Streamlit Authenticator (paso a paso)

Este procedimiento explica, de principio a fin, cómo preparar `config_users.yaml` para
que el inicio de sesión funcione usando [streamlit-authenticator]. Está pensado para el
repositorio actual y se puede repetir cada vez que quieras actualizar usuarios o
contraseñas.

## 1. Instala dependencias (si aún no lo hiciste)

```bash
pip install -r requirements.txt
```

Esto incluye `streamlit-authenticator`, necesario para generar hashes bcrypt.

## 2. Genera hashes seguros para tus contraseñas

Tienes dos opciones:

### Opción A — Modo interactivo (recomendado)

1. Ejecuta el script auxiliar:
   ```bash
   python tools/hash_streamlit_passwords.py --interactive
   ```
2. Escribe la contraseña que quieras asignar a un usuario y presiona Enter.
3. Repite la contraseña cuando se te pida (el script valida que coincidan).
4. Si necesitas generar más hashes, repite los pasos 2 y 3.
5. Presiona Enter sin escribir nada para terminar.
6. Copia cada hash generado (son cadenas largas que empiezan con `$2b$`).

### Opción B — Lista de contraseñas como argumentos (solo para pruebas)

```bash
python tools/hash_streamlit_passwords.py "ClaveUsuario1" "OtraClaveFuerte2025!"
```

Esta modalidad deja las contraseñas en el historial de tu shell, por lo que **no es
adecuada para un entorno real**. Úsala únicamente en local y elimina el historial si es
necesario.

## 3. Actualiza `config_users.yaml`

1. Abre el archivo en tu editor favorito.
2. Localiza la sección `credentials.usernames`.
3. Para cada usuario, reemplaza el valor de `password` por el hash generado
   correspondiente. Ejemplo:

   ```yaml
   juan:
     email: juan@example.com
     name: Juan Test
     password: "$2b$12$Kf5Qx0YJq6b...hash_de_60_caracteres...Xw2P7m4c2"
   ```

4. Mantén las comillas dobles alrededor del hash para evitar errores de parseo.
5. (Opcional pero recomendado) Actualiza `cookie.key` y `cookie.name` con valores nuevos
   y largos para invalidar sesiones anteriores.

## 4. Limpia cookies antiguas

- Borra manualmente las cookies del dominio `*.streamlit.app` en tu navegador **o**
  abre la app en una ventana de incógnito.
- Cambiar `cookie.name` en el paso anterior también fuerza a que las sesiones previas
  se invaliden automáticamente.

## 5. Reinicia la aplicación

- En local: detén `streamlit run app.py` (Ctrl+C) y vuelve a ejecutarlo.
- En Streamlit Cloud: realiza un **Deploy** (o pulsa "Rerun" si ya está desplegada).

## 6. Verifica el inicio de sesión

1. Navega a la app.
2. Introduce el nombre de usuario (`juan`, `ana`, etc.) y la contraseña en texto plano
   que utilizaste para generar el hash correspondiente.
3. Deberías ver el contenido protegido después de iniciar sesión.

## 7. ¿Olvidaste la contraseña?

Si no recuerdas la contraseña original, simplemente repite el procedimiento desde el
paso 2 con una nueva clave. No es necesario conocer la contraseña actual para
reemplazar el hash.

---

[streamlit-authenticator]: https://github.com/mkhorasani/Streamlit-Authenticator
