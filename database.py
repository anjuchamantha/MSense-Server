from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

production = False

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
if not production:
    SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/msense"
else:
    SQLALCHEMY_DATABASE_URL = "postgresql://lhbtxvonfonvom:94daeaffda54ce25e3d0593e3d6f2afbe77c6c9c632de6db55f0ab1a3b7a3f8f@ec2-54-85-113-73.compute-1.amazonaws.com:5432/d55c6ihh26dsso"
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
# )
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
