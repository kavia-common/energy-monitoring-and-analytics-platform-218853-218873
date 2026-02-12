"""
Energy Monitor Backend API (FastAPI).

Provides:
- JWT authentication (signup/login/me)
- Device management (CRUD)
- Energy readings ingestion + query
- Analytics summary endpoint
- Alert rules (threshold per device) CRUD
- WebSocket endpoint for live reading updates

Environment variables expected (set via .env, do not hardcode):
- POSTGRES_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT (DB container provides these)
- JWT_SECRET (required)
- CORS_ALLOW_ORIGINS (optional, comma-separated)
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from passlib.context import CryptContext
import jwt


openapi_tags = [
    {"name": "Health", "description": "Service health and documentation helpers."},
    {"name": "Auth", "description": "User registration, login, and session identity."},
    {"name": "Devices", "description": "Manage energy devices (smart plugs/meters)."},
    {"name": "Readings", "description": "Ingest and query energy readings."},
    {"name": "Analytics", "description": "Aggregations and summary metrics."},
    {"name": "Alerts", "description": "Threshold alert configuration."},
    {"name": "Realtime", "description": "WebSocket live updates."},
]

app = FastAPI(
    title="Energy Monitoring & Analytics API",
    description="Backend for user-scoped device management, energy readings, analytics, alerts, and realtime updates.",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# CORS: allow configured origins; default to wildcard for template friendliness.
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = ["*"] if cors_origins.strip() == "*" else [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database (SQLAlchemy) ---

POSTGRES_URL = os.getenv("POSTGRES_URL")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

def build_db_url() -> str:
    """
    Build SQLAlchemy DB URL from available env vars.
    Prefer POSTGRES_URL if present; otherwise assemble from the individual parts.
    """
    if POSTGRES_URL:
        return POSTGRES_URL
    if not (POSTGRES_USER and POSTGRES_PASSWORD and POSTGRES_DB and POSTGRES_PORT):
        raise RuntimeError(
            "Database env vars not configured. Expected POSTGRES_URL or "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT."
        )
    # Host is local in the multi-container environment.
    return f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"

DATABASE_URL = build_db_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: secrets.token_hex(16))
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    devices: Mapped[List["Device"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    alerts: Mapped[List["AlertRule"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Device(Base):
    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: secrets.token_hex(16))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped[User] = relationship(back_populates="devices")
    readings: Mapped[List["EnergyReading"]] = relationship(back_populates="device", cascade="all, delete-orphan")

class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: secrets.token_hex(16))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    device_id: Mapped[str] = mapped_column(String, ForeignKey("devices.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    watts: Mapped[float] = mapped_column(Float)
    voltage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    device: Mapped[Device] = relationship(back_populates="readings")

class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: secrets.token_hex(16))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    device_id: Mapped[str] = mapped_column(String, ForeignKey("devices.id"), index=True)
    threshold_watts: Mapped[float] = mapped_column(Float)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped[User] = relationship(back_populates="alerts")

def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)

init_db()

def get_db() -> Session:
    """FastAPI dependency to yield a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Auth (JWT) ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    # Keep backend booting in dev, but make it explicit this should be set.
    JWT_SECRET = "DEV_ONLY_CHANGE_ME"

JWT_ALG = "HS256"
JWT_TTL_MIN = 60 * 24  # 24h

def create_access_token(user_id: str, email: str) -> str:
    """Create signed JWT token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_TTL_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain, hashed)

def hash_password(plain: str) -> str:
    """Hash a password."""
    return pwd_context.hash(plain)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Resolve current user from JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Realtime connections manager ---

class ConnectionManager:
    """Tracks per-user websocket connections for broadcasting live readings."""
    def __init__(self) -> None:
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)

    def disconnect(self, user_id: str, websocket: WebSocket) -> None:
        lst = self._connections.get(user_id, [])
        if websocket in lst:
            lst.remove(websocket)
        if not lst and user_id in self._connections:
            del self._connections[user_id]

    async def broadcast(self, user_id: str, message: Dict[str, Any]) -> None:
        for ws in list(self._connections.get(user_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                # Drop broken sockets.
                self.disconnect(user_id, ws)

manager = ConnectionManager()

# --- Schemas ---

class HealthResponse(BaseModel):
    message: str = Field(..., description="Service health message")

class SignupRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password (min 8 chars)")

class LoginResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")

class UserResponse(BaseModel):
    id: str = Field(..., description="User id")
    email: str = Field(..., description="User email")

class DeviceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Device name")
    location: Optional[str] = Field(None, description="Optional location tag")

class DeviceResponse(BaseModel):
    id: str
    name: str
    location: Optional[str] = None
    created_at: datetime

class ReadingCreateRequest(BaseModel):
    device_id: str = Field(..., description="Device id")
    ts: Optional[datetime] = Field(None, description="Timestamp (UTC); defaults to now")
    watts: float = Field(..., ge=0, description="Instantaneous power in watts")
    voltage: Optional[float] = Field(None, ge=0, description="Voltage")
    current: Optional[float] = Field(None, ge=0, description="Current")

class ReadingResponse(BaseModel):
    id: str
    device_id: str
    ts: datetime
    watts: float

class SummaryResponse(BaseModel):
    devices_count: int = Field(..., description="Total devices for the user")
    kwh_24h: float = Field(..., description="Estimated energy usage over last 24h (kWh)")
    current_watts: float = Field(..., description="Most recent reading watts (or 0)")
    alerts_enabled_count: int = Field(..., description="Count of enabled alerts")

class AlertCreateRequest(BaseModel):
    device_id: str = Field(..., description="Device id")
    threshold_watts: float = Field(..., gt=0, description="Threshold watts")
    enabled: bool = Field(True, description="Whether the alert is enabled")

class AlertResponse(BaseModel):
    id: str
    device_id: str
    threshold_watts: float
    enabled: bool
    created_at: datetime

# --- Routes ---

# PUBLIC_INTERFACE
@app.get("/", response_model=HealthResponse, tags=["Health"], summary="Health check")
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(message="Healthy")

# PUBLIC_INTERFACE
@app.get("/docs/ws", tags=["Health"], summary="WebSocket usage help")
def websocket_help() -> Dict[str, Any]:
    """
    WebSocket usage help.

    Connect to:
    - GET ws://<host>/ws?token=<JWT>

    Messages:
    - Server emits {type: 'reading', data: {...}} when new readings are ingested.
    """
    return {
        "ws_url": "/ws?token=<JWT>",
        "message_types": ["reading"],
    }

# PUBLIC_INTERFACE
@app.post("/auth/signup", response_model=UserResponse, tags=["Auth"], summary="Create a new user")
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> UserResponse:
    """Create a user (email + password)."""
    existing = db.scalar(select(User).where(User.email == payload.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserResponse(id=user.id, email=user.email)

# PUBLIC_INTERFACE
@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"], summary="Login and receive a JWT")
def login(payload: SignupRequest, db: Session = Depends(get_db)) -> LoginResponse:
    """Login and return access token."""
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id, user.email)
    return LoginResponse(access_token=token, token_type="bearer")

# PUBLIC_INTERFACE
@app.get("/auth/me", response_model=UserResponse, tags=["Auth"], summary="Get current user")
def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return current user identity."""
    return UserResponse(id=user.id, email=user.email)

# PUBLIC_INTERFACE
@app.get("/devices", response_model=List[DeviceResponse], tags=["Devices"], summary="List devices")
def list_devices(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> List[DeviceResponse]:
    """List devices for the authenticated user."""
    rows = db.scalars(select(Device).where(Device.user_id == user.id).order_by(Device.created_at.desc())).all()
    return [DeviceResponse(id=d.id, name=d.name, location=d.location, created_at=d.created_at) for d in rows]

# PUBLIC_INTERFACE
@app.post("/devices", response_model=DeviceResponse, tags=["Devices"], summary="Create device")
def create_device(
    payload: DeviceCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeviceResponse:
    """Create a new device for the authenticated user."""
    device = Device(user_id=user.id, name=payload.name, location=payload.location)
    db.add(device)
    db.commit()
    db.refresh(device)
    return DeviceResponse(id=device.id, name=device.name, location=device.location, created_at=device.created_at)

# PUBLIC_INTERFACE
@app.delete("/devices/{device_id}", tags=["Devices"], summary="Delete device")
def delete_device(
    device_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete a device owned by the authenticated user."""
    device = db.scalar(select(Device).where(Device.id == device_id, Device.user_id == user.id))
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    db.delete(device)
    db.commit()
    return {"ok": True}

# PUBLIC_INTERFACE
@app.post("/readings", response_model=ReadingResponse, tags=["Readings"], summary="Ingest reading")
async def create_reading(
    payload: ReadingCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReadingResponse:
    """Ingest an energy reading for a device owned by the user."""
    device = db.scalar(select(Device).where(Device.id == payload.device_id, Device.user_id == user.id))
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    ts = payload.ts or datetime.now(timezone.utc)
    reading = EnergyReading(
        user_id=user.id,
        device_id=device.id,
        ts=ts,
        watts=payload.watts,
        voltage=payload.voltage,
        current=payload.current,
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)

    # Broadcast realtime update to user.
    await manager.broadcast(user.id, {"type": "reading", "data": {"device_id": device.id, "ts": reading.ts.isoformat(), "watts": reading.watts}})

    return ReadingResponse(id=reading.id, device_id=reading.device_id, ts=reading.ts, watts=reading.watts)

# PUBLIC_INTERFACE
@app.get("/readings", response_model=List[ReadingResponse], tags=["Readings"], summary="List readings")
def list_readings(
    device_id: Optional[str] = Query(default=None, description="Filter by device id"),
    start: Optional[datetime] = Query(default=None, description="Start timestamp (inclusive)"),
    end: Optional[datetime] = Query(default=None, description="End timestamp (exclusive)"),
    limit: int = Query(default=200, ge=1, le=5000, description="Max number of readings"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[ReadingResponse]:
    """List readings for the authenticated user (optionally filtered)."""
    q = select(EnergyReading).where(EnergyReading.user_id == user.id)
    if device_id:
        q = q.where(EnergyReading.device_id == device_id)
    if start:
        q = q.where(EnergyReading.ts >= start)
    if end:
        q = q.where(EnergyReading.ts < end)
    q = q.order_by(EnergyReading.ts.desc()).limit(limit)

    rows = db.scalars(q).all()
    return [ReadingResponse(id=r.id, device_id=r.device_id, ts=r.ts, watts=r.watts) for r in rows]

# PUBLIC_INTERFACE
@app.get("/analytics/summary", response_model=SummaryResponse, tags=["Analytics"], summary="Get user summary")
def analytics_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> SummaryResponse:
    """Return dashboard summary KPIs."""
    devices_count = db.scalar(select(func.count()).select_from(Device).where(Device.user_id == user.id)) or 0
    alerts_enabled_count = db.scalar(
        select(func.count()).select_from(AlertRule).where(AlertRule.user_id == user.id, AlertRule.enabled.is_(True))
    ) or 0

    # current watts = most recent reading
    latest = db.scalar(
        select(EnergyReading).where(EnergyReading.user_id == user.id).order_by(EnergyReading.ts.desc()).limit(1)
    )
    current_watts = float(latest.watts) if latest else 0.0

    # rough kWh_24h: integrate average watts over period using sample mean and time span.
    now = datetime.now(timezone.utc)
    start_24h = now - timedelta(hours=24)
    rows = db.scalars(
        select(EnergyReading)
        .where(EnergyReading.user_id == user.id, EnergyReading.ts >= start_24h)
        .order_by(EnergyReading.ts.asc())
    ).all()
    if len(rows) < 2:
        kwh_24h = 0.0
    else:
        # Trapezoidal integration in watt-seconds, convert to kWh.
        watt_seconds = 0.0
        for i in range(1, len(rows)):
            dt = (rows[i].ts - rows[i - 1].ts).total_seconds()
            wavg = (rows[i].watts + rows[i - 1].watts) / 2.0
            watt_seconds += max(0.0, dt) * max(0.0, wavg)
        kwh_24h = watt_seconds / 3600.0 / 1000.0

    return SummaryResponse(
        devices_count=int(devices_count),
        kwh_24h=float(kwh_24h),
        current_watts=float(current_watts),
        alerts_enabled_count=int(alerts_enabled_count),
    )

# PUBLIC_INTERFACE
@app.get("/alerts", response_model=List[AlertResponse], tags=["Alerts"], summary="List alerts")
def list_alerts(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> List[AlertResponse]:
    """List alert rules for the authenticated user."""
    rows = db.scalars(select(AlertRule).where(AlertRule.user_id == user.id).order_by(AlertRule.created_at.desc())).all()
    return [
        AlertResponse(
            id=a.id,
            device_id=a.device_id,
            threshold_watts=a.threshold_watts,
            enabled=a.enabled,
            created_at=a.created_at,
        )
        for a in rows
    ]

# PUBLIC_INTERFACE
@app.post("/alerts", response_model=AlertResponse, tags=["Alerts"], summary="Create alert")
def create_alert(
    payload: AlertCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Create threshold alert for a device owned by the user."""
    device = db.scalar(select(Device).where(Device.id == payload.device_id, Device.user_id == user.id))
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    alert = AlertRule(
        user_id=user.id,
        device_id=device.id,
        threshold_watts=payload.threshold_watts,
        enabled=payload.enabled,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return AlertResponse(
        id=alert.id,
        device_id=alert.device_id,
        threshold_watts=alert.threshold_watts,
        enabled=alert.enabled,
        created_at=alert.created_at,
    )

# PUBLIC_INTERFACE
@app.delete("/alerts/{alert_id}", tags=["Alerts"], summary="Delete alert")
def delete_alert(
    alert_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Delete an alert rule owned by the authenticated user."""
    alert = db.scalar(select(AlertRule).where(AlertRule.id == alert_id, AlertRule.user_id == user.id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    db.delete(alert)
    db.commit()
    return {"ok": True}

# PUBLIC_INTERFACE
@app.websocket("/ws", name="Energy live updates")
async def websocket_endpoint(websocket: WebSocket, token: str) -> None:
    """
    WebSocket endpoint for realtime updates.

    Usage:
      ws://<host>/ws?token=<JWT>

    The server will emit:
      { "type": "reading", "data": { "device_id": "...", "ts": "...", "watts": 123 } }
    """
    # Authenticate token manually (OAuth2 dependency doesn't apply to WS).
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=4401)
            return
    except jwt.PyJWTError:
        await websocket.close(code=4401)
        return

    await manager.connect(user_id, websocket)
    try:
        while True:
            # Keep connection open; client may send pings or subscribe messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)
