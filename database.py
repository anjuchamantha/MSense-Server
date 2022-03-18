from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

production = True

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
if not production:
    SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/msense"
else:
    SQLALCHEMY_DATABASE_URL = "postgres://wkruwmsyhblwno:300f39c78c9d2138b07618190e610bd9dc2841a99b59f075804dc30c0c310dc0@ec2-54-144-237-73.compute-1.amazonaws.com:5432/dcb0th0bprbedl"
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
# )
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
