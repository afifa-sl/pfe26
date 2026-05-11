"""
auth.py — Authentification JWT (superadmin / employee)
La BDD users.db est créée automatiquement dans ./data/
"""
import sqlite3, hashlib, secrets, time, os
from typing import Optional
from datetime import datetime

try:
    from config import USERS_DB
except ImportError:
    USERS_DB = "./data/users.db"

TOKEN_TTL = 60 * 60 * 8
ROLES     = ("superadmin", "employee")


def get_conn():
    os.makedirs(os.path.dirname(USERS_DB), exist_ok=True)
    conn = sqlite3.connect(USERS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs("./data", exist_ok=True)
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            nom        TEXT    NOT NULL,
            prenom     TEXT    NOT NULL,
            role       TEXT    NOT NULL DEFAULT 'employee',
            password   TEXT    NOT NULL,
            created_at TEXT    NOT NULL,
            active     INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            expires_at REAL    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS historique (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            question   TEXT    NOT NULL,
            answer     TEXT    NOT NULL,
            source     TEXT    DEFAULT 'rag',
            created_at TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_users_email   ON users(email);
        CREATE INDEX IF NOT EXISTS idx_hist_user     ON historique(user_id);
    """)

    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        conn.execute(
            "INSERT INTO users (email,nom,prenom,role,password,created_at) VALUES (?,?,?,?,?,?)",
            ("admin@sonatrach.dz", "ADMIN", "Super", "superadmin",
             hash_password("Admin@1234"), datetime.now().isoformat())
        )
        conn.commit()
        print("✓ Superadmin créé : admin@sonatrach.dz / Admin@1234")
    else:
        print(f"✓ Users DB chargée ({count} utilisateurs)")
    conn.commit()
    conn.close()


def hash_password(p: str) -> str:
    return hashlib.sha256(f"sonatrach2024{p}".encode()).hexdigest()

def verify_password(p: str, h: str) -> bool:
    return hash_password(p) == h

def create_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
    conn.execute("INSERT INTO sessions VALUES (?,?,?)", (token, user_id, time.time() + TOKEN_TTL))
    conn.commit()
    conn.close()
    return token

def verify_token(token: str) -> Optional[dict]:
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        "SELECT u.id,u.email,u.nom,u.prenom,u.role,u.active,s.expires_at "
        "FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?", (token,)
    ).fetchone()
    conn.close()
    if not row or row["expires_at"] < time.time() or not row["active"]:
        return None
    return dict(row)

def revoke_token(token: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()

def login(email: str, password: str) -> Optional[dict]:
    conn = get_conn()
    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND active=1", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if not user or not verify_password(password, user["password"]):
        return None
    return {
        "token": create_token(user["id"]),
        "user": {
            "id":     user["id"],
            "email":  user["email"],
            "nom":    user["nom"],
            "prenom": user["prenom"],
            "role":   user["role"],
        }
    }

def create_user(email, nom, prenom, role, password):
    if role not in ROLES:
        raise ValueError(f"Rôle invalide: {ROLES}")
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (email,nom,prenom,role,password,created_at) VALUES (?,?,?,?,?,?)",
            (email.lower().strip(), nom.upper().strip(), prenom.strip(),
             role, hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        uid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return {"id": uid, "email": email, "nom": nom, "prenom": prenom, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError(f"Email '{email}' déjà utilisé.")
    finally:
        conn.close()

def list_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id,email,nom,prenom,role,active,created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_user(user_id, **kwargs):
    allowed = {"nom", "prenom", "role", "active", "password"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if "password" in updates:
        updates["password"] = hash_password(updates["password"])
    if not updates:
        return False
    conn = get_conn()
    conn.execute(
        f"UPDATE users SET {','.join(f'{k}=?' for k in updates)} WHERE id=?",
        list(updates.values()) + [user_id]
    )
    conn.commit()
    conn.close()
    return True

def delete_user(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.execute("UPDATE users SET active=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


# ── Historique ────────────────────────────────────────────────────────────────
def save_history(user_id: int, question: str, answer: str, source: str = "rag"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO historique (user_id, question, answer, source, created_at) VALUES (?,?,?,?,?)",
        (user_id, question, answer, source, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(user_id: int, limit: int = 50) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, question, answer, source, created_at FROM historique "
        "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_history(user_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM historique WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
