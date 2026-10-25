import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, select, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy import create_engine

# ---------------- Configuration ----------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///./uptime.db"
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------- Database models ----------------

class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    url: Mapped[str] = mapped_column(String(2048), unique=True)
    interval_seconds: Mapped[int] = mapped_column(Integer, default=30)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )


class Check(Base):
    __tablename__ = "checks"

    id: Mapped[int] = mapped_column(primary_key=True)
    site_id: Mapped[int] = mapped_column(
        ForeignKey("sites.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(20))
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )


# ---------------- Request validation ----------------

class SiteCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: HttpUrl
    interval_seconds: int = Field(default=30, ge=10, le=86400)


# ---------------- Database dependency ----------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------- Website checker ----------------

async def check_website(
    client: httpx.AsyncClient,
    url: str,
) -> dict:
    start = time.perf_counter()

    try:
        response = await client.get(url)
        elapsed = round((time.perf_counter() - start) * 1000, 2)

        if response.status_code >= 500:
            result_status = "DOWN"
        elif response.status_code >= 400:
            result_status = "DEGRADED"
        elif elapsed >= 2000:
            result_status = "DEGRADED"
        else:
            result_status = "UP"

        return {
            "status": result_status,
            "status_code": response.status_code,
            "response_time_ms": elapsed,
            "error": None,
        }

    except (httpx.HTTPError, TimeoutError) as exc:
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": "DOWN",
            "status_code": None,
            "response_time_ms": elapsed,
            "error": str(exc)[:500],
        }


async def save_check(site_id: int, url: str, client: httpx.AsyncClient):
    result = await check_website(client, url)

    # Use a short-lived database session for each saved result.
    with SessionLocal() as db:
        db.add(
            Check(
                site_id=site_id,
                status=result["status"],
                status_code=result["status_code"],
                response_time_ms=result["response_time_ms"],
                error=result["error"],
                checked_at=utc_now(),
            )
        )
        db.commit()

    logger.info(
        "Website check completed: site_id=%s status=%s",
        site_id,
        result["status"],
    )


async def monitor_loop():
    """Run each registered site's checks at its configured interval."""
    next_check: dict[int, float] = {}
    timeout = httpx.Timeout(5.0)
    semaphore = asyncio.Semaphore(10)

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        while True:
            try:
                with SessionLocal() as db:
                    sites = db.scalars(select(Site)).all()
                    site_data = [
                        (site.id, site.url, site.interval_seconds)
                        for site in sites
                    ]

                now = time.monotonic()
                active_ids = {site_id for site_id, _, _ in site_data}

                # Forget deleted sites.
                next_check = {
                    site_id: due
                    for site_id, due in next_check.items()
                    if site_id in active_ids
                }

                async def limited_check(site_id, url):
                    async with semaphore:
                        await save_check(site_id, url, client)

                tasks = []

                for site_id, url, interval in site_data:
                    if now >= next_check.get(site_id, 0):
                        next_check[site_id] = now + interval
                        tasks.append(limited_check(site_id, url))

                if tasks:
                    results = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error("Monitoring task failed: %s", result)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Monitoring loop failed")

            await asyncio.sleep(2)


# ---------------- Application lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    task = asyncio.create_task(monitor_loop())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Mini Uptime Monitor",
    description="Monitor website availability and response times.",
    lifespan=lifespan,
)


# ---------------- Dashboard ----------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# ---------------- REST API ----------------

@app.post("/api/sites", status_code=201)
def add_site(data: SiteCreate, db: Session = Depends(get_db)):
    url = str(data.url)

    if data.name.strip() == "":
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    existing = db.scalar(select(Site).where(Site.url == url))
    if existing:
        raise HTTPException(
            status_code=409,
            detail="This website is already registered",
        )

    site = Site(
        name=data.name.strip(),
        url=url,
        interval_seconds=data.interval_seconds,
    )
    db.add(site)
    db.commit()
    db.refresh(site)

    return {
        "id": site.id,
        "name": site.name,
        "url": site.url,
        "interval_seconds": site.interval_seconds,
        "message": "Website registered successfully",
    }


@app.get("/api/sites")
def list_sites(db: Session = Depends(get_db)):
    sites = db.scalars(select(Site).order_by(Site.id.desc())).all()
    output = []

    for site in sites:
        latest = db.scalar(
            select(Check)
            .where(Check.site_id == site.id)
            .order_by(Check.checked_at.desc())
            .limit(1)
        )

        total = db.scalar(
            select(func.count(Check.id)).where(Check.site_id == site.id)
        ) or 0

        up_count = db.scalar(
            select(func.count(Check.id)).where(
                Check.site_id == site.id,
                Check.status == "UP",
            )
        ) or 0

        output.append({
            "id": site.id,
            "name": site.name,
            "url": site.url,
            "interval_seconds": site.interval_seconds,
            "status": latest.status if latest else "PENDING",
            "status_code": latest.status_code if latest else None,
            "response_time_ms": (
                latest.response_time_ms if latest else None
            ),
            "last_checked": (
                latest.checked_at.isoformat() if latest else None
            ),
            "uptime_percentage": (
                round(up_count / total * 100, 2) if total else None
            ),
        })

    return output


@app.get("/api/sites/{site_id}/checks")
def site_history(
    site_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    if db.get(Site, site_id) is None:
        raise HTTPException(status_code=404, detail="Website not found")

    limit = max(1, min(limit, 200))
    checks = db.scalars(
        select(Check)
        .where(Check.site_id == site_id)
        .order_by(Check.checked_at.desc())
        .limit(limit)
    ).all()

    return [
        {
            "id": check.id,
            "status": check.status,
            "status_code": check.status_code,
            "response_time_ms": check.response_time_ms,
            "error": check.error,
            "checked_at": check.checked_at.isoformat(),
        }
        for check in checks
    ]


@app.post("/api/sites/{site_id}/check")
async def check_now(site_id: int, db: Session = Depends(get_db)):
    site = db.get(Site, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Website not found")

    async with httpx.AsyncClient(
        timeout=5.0,
        follow_redirects=True,
    ) as client:
        result = await check_website(client, site.url)

    record = Check(
        site_id=site.id,
        status=result["status"],
        status_code=result["status_code"],
        response_time_ms=result["response_time_ms"],
        error=result["error"],
        checked_at=utc_now(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "site_id": site.id,
        "url": site.url,
        **result,
        "checked_at": record.checked_at.isoformat(),
    }


@app.delete("/api/sites/{site_id}")
def delete_site(site_id: int, db: Session = Depends(get_db)):
    site = db.get(Site, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Website not found")

    db.delete(site)
    db.commit()
    return {"message": "Website deleted successfully"}