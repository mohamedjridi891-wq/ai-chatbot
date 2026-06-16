from pathlib import Path
from datetime import datetime
import hashlib
import zipfile
try:
    import rarfile
except ImportError:
    rarfile = None
import logging
import os
import shutil
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ROOT_FOLDER = Path(os.getenv("ROOT_FOLDER", BASE_DIR / "root"))
DB_URL = os.getenv("DB_URL")
REVIEW_FOLDER = Path(os.getenv("REVIEW_FOLDER", BASE_DIR / "review"))

REVIEW_CATEGORIES = {
    "Uncategorized", "VirtualMachine", "Backup", "Log",
    "Temp", "CAD", "3D", "Font", "Audio", "Video",
}

FILE_TYPES = {
    # Documents
    ".pdf": "Document",     ".docx": "Document",    ".doc": "Document",
    ".txt": "Document",     ".rtf": "Document",     ".odt": "Document",
    ".md": "Document",      ".tex": "Document",     ".wps": "Document",
    ".wpd": "Document",     ".pages": "Document",   ".epub": "Document",
    ".mobi": "Document",    ".djvu": "Document",    ".wri": "Document",
    ".cnt": "Document",     ".diz": "Document",     ".nfo": "Document",
    ".onetoc2": "Document", ".license": "Document",

    # Presentations
    ".pptx": "Presentation", ".ppt": "Presentation", ".odp": "Presentation",
    ".key": "Presentation",  ".ppsx": "Presentation", ".pps": "Presentation",

    # Spreadsheets
    ".xlsx": "Spreadsheet", ".xls": "Spreadsheet",  ".csv": "Spreadsheet",
    ".ods": "Spreadsheet",  ".tsv": "Spreadsheet",  ".numbers": "Spreadsheet",
    ".xlsm": "Spreadsheet",

    # Images
    ".jpg": "Image",        ".jpeg": "Image",       ".png": "Image",
    ".gif": "Image",        ".bmp": "Image",        ".tiff": "Image",
    ".tif": "Image",        ".webp": "Image",       ".svg": "Image",
    ".ico": "Image",        ".heic": "Image",       ".heif": "Image",
    ".raw": "Image",        ".cr2": "Image",        ".nef": "Image",
    ".psd": "Image",        ".ai": "Image",         ".eps": "Image",
    ".xcf": "Image",        ".cur": "Image",        ".ani": "Image",
    ".skin": "Image",       ".hdr": "Image",        ".psp": "Image",
    ".icm": "Image",        ".icc": "Image",        ".cel": "Image",
    ".aco": "Image",        ".acb": "Image",        ".acv": "Image",
    ".atn": "Image",        ".grd": "Image",        ".pat": "Image",
    ".abr": "Image",        ".asl": "Image",        ".ahu": "Image",
    ".alv": "Image",        ".blw": "Image",        ".pqg": "Image",
    ".csf": "Image",        ".dbrush": "Image",     ".dbrsb": "Image",
    ".8bf": "Image",        ".8bi": "Image",        ".8bx": "Image",
    ".8ba": "Image",        ".8be": "Image",        ".8me": "Image",
    ".8li": "Image",        ".exv": "Image",

    # Video
    ".mp4": "Video",        ".avi": "Video",        ".mov": "Video",
    ".mkv": "Video",        ".wmv": "Video",        ".flv": "Video",
    ".webm": "Video",       ".m4v": "Video",        ".mpeg": "Video",
    ".mpg": "Video",        ".3gp": "Video",        ".ts": "Video",
    ".vob": "Video",        ".rmvb": "Video",

    # Audio
    ".mp3": "Audio",        ".wav": "Audio",        ".flac": "Audio",
    ".aac": "Audio",        ".ogg": "Audio",        ".wma": "Audio",
    ".m4a": "Audio",        ".opus": "Audio",       ".aiff": "Audio",
    ".mid": "Audio",        ".midi": "Audio",       ".au": "Audio",

    # Archives
    ".zip": "Archive",      ".rar": "Archive",      ".7z": "Archive",
    ".tar": "Archive",      ".gz": "Archive",       ".bz2": "Archive",
    ".xz": "Archive",       ".cab": "Archive",      ".iso": "Archive",
    ".tgz": "Archive",      ".z": "Archive",

    # Executables & Installers
    ".exe": "Executable",   ".msi": "Executable",   ".apk": "Executable",
    ".app": "Executable",   ".dmg": "Executable",   ".deb": "Executable",
    ".rpm": "Executable",   ".run": "Executable",   ".bat": "Executable",
    ".cmd": "Executable",   ".com": "Executable",   ".swf": "Executable",
    ".msu": "Executable",   ".msp": "Executable",   ".pkg": "Executable",
    ".comppk": "Executable",

    # System & Binary
    ".dll": "System",       ".sys": "System",       ".drv": "System",
    ".so": "System",        ".bin": "System",       ".dat": "System",
    ".img": "System",       ".rom": "System",       ".bios": "System",
    ".efi": "System",       ".mui": "System",       ".vxd": "System",
    ".ovl": "System",       ".hlp": "System",       ".chm": "System",
    ".ntf": "System",       ".mnu": "System",       ".cpi": "System",
    ".pif": "System",       ".mbr": "System",       ".vfd": "System",
    ".sdb": "System",       ".hiv": "System",       ".dos": "System",
    ".icl": "System",       ".inx": "System",       ".frp": "System",
    ".mdl": "System",       ".gll": "System",       ".ilk": "System",
    ".binary": "System",    ".ncb": "System",       ".pdb": "System",
    ".tlb": "System",       ".lib": "System",       ".exp": "System",
    ".dl_": "System",       ".ex_": "System",       ".nl_": "System",
    ".sy_": "System",       ".fo_": "System",       ".ap_": "System",
    ".bin_sts": "System",   ".irs": "System",       ".csh": "Code",

    # Code — Web
    ".html": "Code",        ".htm": "Code",         ".css": "Code",
    ".js": "Code",          ".jsx": "Code",         ".tsx": "Code",
    ".php": "Code",         ".asp": "Code",         ".aspx": "Code",
    ".vue": "Code",         ".svelte": "Code",      ".scss": "Code",
    ".less": "Code",        ".xsl": "Code",         ".xsd": "Code",
    ".tld": "Code",         ".xhtml": "Code",       ".shtml": "Code",
    ".coffee": "Code",      ".ashx": "Code",        ".master": "Code",
    ".webinfo": "Code",     ".csproj": "Code",      ".resx": "Code",
    ".script": "Code",

    # Code — General
    ".py": "Code",          ".java": "Code",        ".c": "Code",
    ".cpp": "Code",         ".cc": "Code",          ".cxx": "Code",
    ".h": "Code",           ".hpp": "Code",         ".cs": "Code",
    ".go": "Code",          ".rs": "Code",          ".rb": "Code",
    ".swift": "Code",       ".kt": "Code",          ".scala": "Code",
    ".r": "Code",           ".m": "Code",           ".pl": "Code",
    ".lua": "Code",         ".dart": "Code",        ".vb": "Code",
    ".asm": "Code",         ".f90": "Code",         ".pas": "Code",

    # Code — Shell & Scripts
    ".sh": "Code",          ".bash": "Code",        ".zsh": "Code",
    ".fish": "Code",        ".ps1": "Code",         ".psm1": "Code",
    ".vbs": "Code",

    # Java & Android
    ".jar": "Code",         ".class": "Code",       ".dex": "Code",
    ".aidl": "Code",        ".war": "Code",         ".ejb": "Code",
    ".gradle": "Code",      ".pom": "Code",         ".mf": "Code",
    ".jsp": "Code",         ".jspx": "Code",        ".jspf": "Code",

    # .NET & Visual Studio
    ".sln": "Code",

    # Data
    ".json": "Data",        ".xml": "Data",         ".yaml": "Data",
    ".yml": "Data",         ".toml": "Data",        ".parquet": "Data",
    ".xyz": "Data",         ".tab": "Data",         ".ado": "Data",
    ".md5": "Data",         ".rpt": "Data",         ".lst": "Data",
    ".ins": "Data",         ".lid": "Data",         ".sif": "Data",
    ".vce": "Data",         ".dir": "Data",         ".act": "Data",
    ".zvt": "Data",         ".p3r": "Data",         ".p3l": "Data",
    ".p3m": "Data",         ".p3e": "Data",         ".ple": "Data",
    ".iros": "Data",        ".slc": "Data",         ".rtc": "Data",
    ".eap": "Data",         ".ilm": "Data",         ".flp": "Data",
    ".eve": "Data",         ".cha": "Data",         ".hdt": "Data",
    ".whs": "Data",         ".apl": "Data",         ".mfm": "Data",
    ".shc": "Data",         ".kys": "Data",         ".rw": "Data",
    ".enumerated": "Data",  ".sp3": "Data",

    # Config
    ".ini": "Config",       ".cfg": "Config",       ".conf": "Config",
    ".env": "Config",       ".cfg": "Config",       ".conf": "Config",
    ".env": "Config",       ".properties": "Config", ".reg": "Config",
    ".plist": "Config",     ".config": "Config",    ".prefs": "Config",
    ".policy": "Config",    ".manifest": "Config",  ".inf": "Config",
    ".adm": "Config",       ".pro": "Config",       ".lang": "Config",
    ".strings": "Config",   ".store": "Config",     ".info": "Config",
    ".sitemap": "Config",   ".jdbc": "Config",      ".tpl": "Config",
    ".map": "Config",       ".tag": "Config",       ".refresh": "Config",
    ".lookup": "Config",    ".mappings": "Config",  ".access": "Config",
    ".marker": "Config",    ".component": "Config", ".exsd": "Config",
    ".e4xmi": "Config",     ".iml": "Config",       ".suo": "Config",
    ".vsmdi": "Config",     ".vspscc": "Config",    ".vssscc": "Config",
    ".user": "Config",      ".settings": "Config",  ".ppd": "Config",
    ".slf": "Config",       ".lock": "Config",      ".exclude": "Config",
    ".oem": "Config",       ".pdsync": "Config",    ".pdsyncu": "Config",

    # Database
    ".sql": "Database",     ".db": "Database",      ".sqlite": "Database",
    ".sqlite3": "Database", ".mdb": "Database",     ".accdb": "Database",
    ".dbf": "Database",     ".mdf": "Database",     ".ldf": "Database",
    ".sdf": "Database",     ".wdb": "Database",     ".dbd": "Database",

    # Fonts
    ".ttf": "Font",         ".otf": "Font",         ".woff": "Font",
    ".woff2": "Font",       ".eot": "Font",         ".fon": "Font",

    # 3D & CAD
    ".obj": "3D",           ".fbx": "3D",           ".stl": "3D",
    ".blend": "3D",         ".dae": "3D",           ".3ds": "3D",
    ".dwg": "CAD",          ".dxf": "CAD",          ".step": "CAD",
    ".stp": "CAD",          ".iges": "CAD",         ".drw": "CAD",

    # Temp & Logs
    ".tmp": "Temp",         ".temp": "Temp",        ".cache": "Temp",
    ".log": "Log",

    # Backup
    ".bak": "Backup",       ".old": "Backup",       ".orig": "Backup",
    ".v20150204-1700": "Backup", ".201503100339-release-e44": "Backup",
    ".24022014": "Backup",

    # Shortcuts
    ".lnk": "Shortcut",     ".url": "Shortcut",     ".desktop": "Shortcut",
    ".webloc": "Shortcut",

    # Virtual Machines
    ".vmdk": "VirtualMachine", ".vhd": "VirtualMachine", ".vhdx": "VirtualMachine",
    ".ova": "VirtualMachine",  ".ovf": "VirtualMachine", ".qcow2": "VirtualMachine",

    # Security & Certificates
    ".pem": "Security",     ".crt": "Security",     ".cer": "Security",
    ".key": "Security",     ".p12": "Security",     ".pfx": "Security",
    ".pub": "Security",     ".rsa": "Security",     ".sf": "Security",
    ".pk8": "Security",     ".der": "Security",     ".keystore": "Security",
    ".password": "Security",
}

CRITICAL_EXTENSIONS = {
    '.docx', '.doc', '.pdf', '.rtf', '.odt', '.wpd', '.wps', '.tex',
    '.xlsx', '.xls', '.ods', '.numbers', '.xlsm',
    '.pptx', '.ppt', '.odp', '.key', '.ppsx', '.pps',
    '.sql', '.db', '.sqlite', '.sqlite3', '.mdb', '.accdb', '.dbf',
    '.mdf', '.ldf', '.sdf',
    '.pem', '.crt', '.cer', '.p12', '.pfx', '.pub', '.keystore',
}

TEMP_EXTENSIONS = {
    '.tmp', '.temp', '.cache', '.old', '.orig',
    '.log', '.bak', '.md5',
    '.ds_store', '.thumbs',
    '.lnk', '.desktop', '.url', '.webloc',
    '.suo', '.ilk', '.lock',
}

ARCHIVE_EXTENSIONS = {
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
    '.xz', '.cab', '.iso', '.tgz', '.z',
}

CSV_FIELDS = [
    "name", "stem", "extension", "category", "path", "folder",
    "depth", "size_mb", "size_bytes", "is_empty",
    "created_time", "modified_time", "access_time",
    "hash", "status", "source_archive",
]

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "hash_errors.log"),
    level=logging.WARNING,
    format="%(asctime)s — %(message)s"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def get_db_connection():
    if not DB_URL:
        raise RuntimeError("DB_URL not set. Please set DB_URL in your .env or environment variables.")
    return psycopg2.connect(DB_URL)


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id SERIAL PRIMARY KEY,
                name TEXT,
                stem TEXT,
                extension TEXT,
                category TEXT,
                path TEXT UNIQUE,
                folder TEXT,
                depth INTEGER,
                size_mb REAL,
                size_bytes BIGINT,
                is_empty BOOLEAN,
                created_time TIMESTAMP,
                modified_time TIMESTAMP,
                access_time TIMESTAMP,
                hash TEXT,
                status TEXT,
                source_archive TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);")
    conn.commit()


def insert_rows(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO files (
            name, stem, extension, category, path, folder,
            depth, size_mb, size_bytes, is_empty,
            created_time, modified_time, access_time,
            hash, status, source_archive
        ) VALUES %s
        ON CONFLICT (path) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()


def get_category(ext: str) -> str:
    return FILE_TYPES.get(ext, "Uncategorized")


def get_status(ext: str, size_bytes: int) -> str:
    if size_bytes == 0:
        return "DELETE_CANDIDATE"
    if ext in TEMP_EXTENSIONS:
        return "DELETE_CANDIDATE"
    if ext in CRITICAL_EXTENSIONS:
        return "KEEP"
    if ext in ARCHIVE_EXTENSIONS:
        return "ARCHIVE"
    return "REVIEW"


def hash_file(filepath) -> str:
    try:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except PermissionError:
        logging.warning(f"Permission denied: {filepath}")
        return "error_permission"
    except FileNotFoundError:
        logging.warning(f"Not found (deleted during scan?): {filepath}")
        return "error_not_found"
    except OSError as e:
        logging.warning(f"OS error on {filepath}: {e}")
        return "error_locked"
    except Exception as e:
        logging.warning(f"Unexpected hash error on {filepath}: {e}")
        return "error_unknown"


def hash_bytes(data: bytes) -> str:
    try:
        return hashlib.sha256(data).hexdigest()
    except Exception as e:
        logging.warning(f"Hash from bytes failed: {e}")
        return "error_unknown"


def move_to_review(file_path: Path, root_path: Path) -> bool:
    try:
        review_root = Path(REVIEW_FOLDER)
        review_root.mkdir(parents=True, exist_ok=True)
        try:
            rel_path = file_path.relative_to(root_path)
        except Exception:
            rel_path = file_path.name
        target = review_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(target))
        print(f"  → Moved to review: {file_path} -> {target}")
        return True
    except Exception as e:
        logging.warning(f"Could not move {file_path} to review: {e}")
        print(f"  ⚠  Could not move to review: {file_path}: {e}")
        return False


def make_row(name, stem, ext, path, folder, depth,
             size_bytes, created, modified, accessed,
             file_hash, source_archive="") -> tuple:
    return (
        name,
        stem,
        ext,
        get_category(ext),
        path,
        folder,
        depth,
        round(size_bytes / (1024 * 1024), 2),
        size_bytes,
        size_bytes == 0,
        created,
        modified,
        accessed,
        file_hash,
        get_status(ext, size_bytes),
        source_archive,
    )


def scan_zip(zip_path: Path, base_depth: int, append_row, counters: dict) -> bool:
    rows = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename)
                ext = inner.suffix.lower()
                dt = datetime(*info.date_time) if info.date_time else datetime.min
                try:
                    data = zf.read(info.filename)
                    h = hash_bytes(data)
                    sz = info.file_size
                except Exception:
                    data, h, sz = b"", "error", 0

                if get_status(ext, sz) == "DELETE_CANDIDATE":
                    continue
                if get_category(ext) in REVIEW_CATEGORIES:
                    continue

                rows.append(make_row(
                    name=inner.name,
                    stem=inner.stem,
                    ext=ext,
                    path=f"{zip_path}::{info.filename}",
                    folder=str(zip_path) + "::" + str(inner.parent),
                    depth=base_depth + len(inner.parts),
                    size_bytes=sz,
                    created=dt,
                    modified=dt,
                    accessed=dt,
                    file_hash=h,
                    source_archive=str(zip_path),
                ))
        for row in rows:
            append_row(row)
            counters["zip"] += 1
        return True
    except zipfile.BadZipFile:
        print(f"  ⚠  Bad/corrupt zip — moved to review: {zip_path}")
        return False
    except Exception as e:
        print(f"  ⚠  Could not read zip {zip_path}: {e}")
        return False


def scan_rar(rar_path: Path, base_depth: int, append_row, counters: dict) -> bool:
    rows = []
    try:
        with rarfile.RarFile(rar_path, "r") as rf:
            for info in rf.infolist():
                if info.is_dir():
                    continue
                inner = Path(info.filename)
                ext = inner.suffix.lower()
                dt = info.mtime if info.mtime else datetime.min
                try:
                    data = rf.read(info.filename)
                    h = hash_bytes(data)
                    sz = info.file_size
                except Exception:
                    data, h, sz = b"", "error", 0

                if get_status(ext, sz) == "DELETE_CANDIDATE":
                    continue
                if get_category(ext) in REVIEW_CATEGORIES:
                    continue

                rows.append(make_row(
                    name=inner.name,
                    stem=inner.stem,
                    ext=ext,
                    path=f"{rar_path}::{info.filename}",
                    folder=str(rar_path) + "::" + str(inner.parent),
                    depth=base_depth + len(inner.parts),
                    size_bytes=sz,
                    created=dt,
                    modified=dt,
                    accessed=dt,
                    file_hash=h,
                    source_archive=str(rar_path),
                ))
        for row in rows:
            append_row(row)
            counters["rar"] += 1
        return True
    except rarfile.BadRarFile:
        print(f"  ⚠  Bad/corrupt rar — moved to review: {rar_path}")
        return False
    except rarfile.NeedFirstVolume:
        print(f"  ⚠  Multi-volume RAR (not first volume) — skipped: {rar_path}")
        return False
    except Exception as e:
        print(f"  ⚠  Could not read rar {rar_path}: {e}")
        return False


def run_phase1(root_folder: str = ROOT_FOLDER):
    print(f"\n{'='*60}")
    print("  Phase 1 — File Inventory")
    print(f"{'='*60}")
    print(f"  Root folder: {root_folder}")

    root = Path(root_folder)
    counters = {"real": 0, "zip": 0, "rar": 0, "deleted": 0, "reviewed": 0}
    total_size = 0.0

    conn = get_db_connection()
    create_tables(conn)

    rows = []

    def append_row(row):
        rows.append(row)
        if len(rows) >= 500:
            insert_rows(conn, rows)
            rows.clear()

    def flush_rows():
        if rows:
            insert_rows(conn, rows)
            rows.clear()

    review_dir = Path(REVIEW_FOLDER).resolve()

    for file in root.rglob("*"):
        if not file.is_file():
            continue

        try:
            stat = file.stat()
        except OSError:
            continue

        try:
            resolved = file.resolve()
        except OSError:
            resolved = file

        if review_dir in resolved.parents or resolved == review_dir:
            continue

        ext = file.suffix.lower()
        size_bytes = stat.st_size
        depth = len(file.relative_to(root).parts) - 1
        category = get_category(ext)
        status = get_status(ext, size_bytes)

        if status == "DELETE_CANDIDATE":
            try:
                file.unlink()
                counters["deleted"] += 1
                print(f"  ✂ Deleted DELETE_CANDIDATE: {file}")
            except Exception as e:
                logging.warning(f"Could not delete {file}: {e}")
                print(f"  ⚠  Could not delete {file}: {e}")
            continue

        if category in REVIEW_CATEGORIES:
            if move_to_review(file, root):
                counters["reviewed"] += 1
            continue

        if ext == ".zip":
            try:
                with zipfile.ZipFile(file, "r"):
                    pass
            except zipfile.BadZipFile:
                if move_to_review(file, root):
                    counters["reviewed"] += 1
                continue
            except Exception as e:
                print(f"  ⚠  Could not open zip {file}: {e}")
                if move_to_review(file, root):
                    counters["reviewed"] += 1
                continue
        elif ext == ".rar":
            if rarfile is not None:
                try:
                    with rarfile.RarFile(file, "r"):
                        pass
                except rarfile.BadRarFile:
                    if move_to_review(file, root):
                        counters["reviewed"] += 1
                    continue
                except rarfile.NeedFirstVolume:
                    print(f"  ⚠  Multi-volume RAR (not first volume) — skipped: {file}")
                    continue
                except Exception as e:
                    print(f"  ⚠  Could not open rar {file}: {e}")
                    if move_to_review(file, root):
                        counters["reviewed"] += 1
                    continue
            else:
                print(f"  ⚠  rarfile module missing, skipping archive content for {file}")

        row = make_row(
            name=file.name,
            stem=file.stem,
            ext=ext,
            path=str(file.resolve()),
            folder=str(file.parent),
            depth=depth,
            size_bytes=size_bytes,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            accessed=datetime.fromtimestamp(stat.st_atime),
            file_hash=hash_file(str(file)),
        )
        append_row(row)
        counters["real"] += 1
        total_size += row[7]

        if ext == ".zip":
            scan_zip(file, depth, append_row, counters)
        elif ext == ".rar":
            if rarfile is not None:
                scan_rar(file, depth, append_row, counters)
            else:
                print(f"  ⚠  rarfile module missing, skipping archive content for {file}")

        if counters["real"] % 500 == 0:
            print(f"  … {counters['real']} files scanned", end="\r")

    flush_rows()
    conn.close()

    total = counters["real"] + counters["zip"] + counters["rar"]
    print(f"\nTotal entries  : {total}")
    print(f"  Real files   : {counters['real']}")
    print(f"  Inside ZIPs  : {counters['zip']}")
    print(f"  Inside RARs  : {counters['rar']}")
    print(f"  Deleted files: {counters['deleted']}")
    print(f"  Moved to review: {counters['reviewed']}")
    print(f"\nTotal size (real files) : {total_size:.2f} MB")
    print("\nOutput → files table")
    print("Phase 1 complete")


if __name__ == "__main__":
    run_phase1()
