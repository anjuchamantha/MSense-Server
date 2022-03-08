from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

production = False

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
if not production:
    SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/msense"
else:
    SQLALCHEMY_DATABASE_URL = "postgresql://pmquekznzhdvjd:91cb91840ee19652f43450225f90c0a3eca0ae02ed80e1df63c8ab6d6c709510@ec2-18-210-191-5.compute-1.amazonaws.com/d2lvlreee2bjqt"

# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
# )
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
